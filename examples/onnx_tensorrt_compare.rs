//! # ONNX vs TensorRT Comparison Example
//!
//! Compares inference results between ONNX Runtime and TensorRT engines
//! to validate that the TensorRT conversion produces equivalent results.
//!
//! ## Usage
//! ```bash
//! cargo run --example onnx_tensorrt_compare -- --onnx /path/to/model.onnx --engine /path/to/model.engine
//! ```
//!
//! ## Requirements
//! - ONNX model file (.onnx)
//! - Corresponding TensorRT engine file (.engine)
//! - Both models should have the same input/output specifications
//!
//! ## What it does
//! 1. Loads both the ONNX model (via ort crate) and TensorRT engine (via libinfer)
//! 2. Generates random input data matching the model's input dimensions
//! 3. Runs inference on both models with the same input
//! 4. Compares the outputs to ensure they are approximately equal
//!
//! This helps verify that TensorRT optimization hasn't changed the model's behavior.

use anyhow::{anyhow, Result};
use clap::Parser;
use cxx::UniquePtr;
use libinfer::{Engine, TensorDataType, Options};
use libinfer::ffi::InputTensor;
use ort::execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{info, debug, Level, warn};
use tracing_subscriber::{FmtSubscriber, EnvFilter};

#[derive(clap::ValueEnum, Clone, Debug)]
enum ExecutionProvider {
    CPU,
    CUDA,
    TensorRT,
}

impl std::fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionProvider::CPU => write!(f, "CPU"),
            ExecutionProvider::CUDA => write!(f, "CUDA"),
            ExecutionProvider::TensorRT => write!(f, "TensorRT"),
        }
    }
}

#[derive(Parser, Debug)]
#[clap(about = "Compare ONNX Runtime vs TensorRT inference results")]
struct Args {
    /// Path to the ONNX model file
    #[arg(long, value_name = "PATH", value_parser)]
    onnx: PathBuf,

    /// ONNX execution provider
    #[arg(long, default_value_t = ExecutionProvider::TensorRT, value_enum)]
    onnx_ep: ExecutionProvider,

    /// Path to the TensorRT engine file
    #[arg(long, value_name = "PATH", value_parser)]
    engine: PathBuf,

    /// GPU device index to use 
    #[arg(long, default_value_t = 0)]
    device: u32,

    /// Tolerance for output comparison 
    #[arg(long, default_value_t = 0.1)]
    tolerance: f32,

    /// Number of random test iterations 
    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// Random seed for reproducible results 
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of benchmark iterations for performance testing 
    #[arg(long, default_value_t = 1000)]
    benchmark_iterations: usize,

    /// Skip performance benchmarking and only do correctness comparison
    #[arg(long, default_value_t = false)]
    skip_benchmark: bool,
}

fn generate_random_input_u8(dims: &[u32], rng: &mut StdRng) -> Vec<u8> {
    let size: usize = dims.iter().map(|&d| d as usize).product();
    (0..size).map(|_| rng.gen_range(0..=255)).collect()
}

fn generate_random_input_f32(dims: &[u32], rng: &mut StdRng) -> Vec<f32> {
    let size: usize = dims.iter().map(|&d| d as usize).product();
    (0..size).map(|_| rng.gen_range(-1.0..=1.0)).collect()
}

fn generate_random_input_int64(dims: &[u32], rng: &mut StdRng) -> Vec<i64> {
    let size: usize = dims.iter().map(|&d| d as usize).product();
    (0..size).map(|_| rng.gen_range(-1000..=1000)).collect()
}

fn generate_random_input_bool(dims: &[u32], rng: &mut StdRng) -> Vec<bool> {
    let size: usize = dims.iter().map(|&d| d as usize).product();
    (0..size).map(|_| rng.gen_bool(0.5)).collect()
}

fn f32_to_u8(data: &[f32]) -> Vec<u8> {
    // Convert f32 values to their byte representation (4 bytes per f32)
    // Use little-endian to match the from_le_bytes() conversion later
    data.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect()
}

fn int64_to_u8(data: &[i64]) -> Vec<u8> {
    // Convert i64 values to their byte representation (8 bytes per i64)
    // Use little-endian to match the from_le_bytes() conversion later
    data.iter()
        .flat_map(|&i| i.to_le_bytes())
        .collect()
}

fn bool_to_u8(data: &[bool]) -> Vec<u8> {
    // Convert bool values to byte representation (1 byte per bool)
    data.iter()
        .map(|&b| if b { 1u8 } else { 0u8 })
        .collect()
}

fn run_onnx_inference_f32(session: &mut Session, input_data_list: &[Vec<f32>], input_infos: &[libinfer::TensorInfo]) -> Result<Vec<Vec<f32>>> {
    // Get output names before mutable borrow
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
    
    // Create input tensors for all inputs (f32 data type)
    let mut input_tensors = Vec::new();
    for (i, input_data) in input_data_list.iter().enumerate() {
        let input_info = &input_infos[i];
        // Create tensor with the shape provided from TensorRT (add batch dimension)
        let shape_with_batch: Vec<usize> = std::iter::once(1usize) // batch size 1
            .chain(input_info.dims.iter().map(|&d| d as usize))
            .collect();
        
        let input_tensor = Value::from_array((shape_with_batch, input_data.clone()))?;
        input_tensors.push((input_info.name.as_str(), input_tensor));
    }
    
    // Run inference with all inputs
    let inputs_map = input_tensors.into_iter().collect::<std::collections::HashMap<_, _>>();
    let outputs = session.run(inputs_map)?;
    
    // Extract all outputs
    let mut result_outputs = Vec::new();
    for output_name in output_names {
        let output_tensor = outputs[output_name.as_str()].try_extract_array::<f32>()?;
        result_outputs.push(output_tensor.to_owned().into_raw_vec_and_offset().0);
    }
    
    Ok(result_outputs)
}

fn run_onnx_inference_u8(session: &mut Session, input_data_list: &[Vec<u8>], input_infos: &[libinfer::TensorInfo]) -> Result<Vec<Vec<f32>>> {
    // Get output names before mutable borrow
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
    
    // Create input tensors for all inputs (u8 data type)
    let mut input_tensors = Vec::new();
    for (i, input_data) in input_data_list.iter().enumerate() {
        let input_info = &input_infos[i];
        // Create tensor with the shape provided from TensorRT (add batch dimension)
        let shape_with_batch: Vec<usize> = std::iter::once(1usize) // batch size 1
            .chain(input_info.dims.iter().map(|&d| d as usize))
            .collect();
        
        let input_tensor = Value::from_array((shape_with_batch, input_data.clone()))?;
        input_tensors.push((input_info.name.as_str(), input_tensor));
    }
    
    // Run inference with all inputs
    let inputs_map = input_tensors.into_iter().collect::<std::collections::HashMap<_, _>>();
    let outputs = session.run(inputs_map)?;
    
    // Extract all outputs
    let mut result_outputs = Vec::new();
    for output_name in output_names {
        let output_tensor = outputs[output_name.as_str()].try_extract_array::<f32>()?;
        result_outputs.push(output_tensor.to_owned().into_raw_vec_and_offset().0);
    }
    
    Ok(result_outputs)
}

fn run_onnx_inference_int64(session: &mut Session, input_data_list: &[Vec<int64>], input_infos: &[libinfer::TensorInfo]) -> Result<Vec<Vec<f32>>> {
    // Get output names before mutable borrow
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
    
    // Create input tensors for all inputs (i64 data type)
    let mut input_tensors = Vec::new();
    for (i, input_data) in input_data_list.iter().enumerate() {
        let input_info = &input_infos[i];
        // Create tensor with the shape provided from TensorRT (add batch dimension)
        let shape_with_batch: Vec<usize> = std::iter::once(1usize) // batch size 1
            .chain(input_info.dims.iter().map(|&d| d as usize))
            .collect();
        
        let input_tensor = Value::from_array((shape_with_batch, input_data.clone()))?;
        input_tensors.push((input_info.name.as_str(), input_tensor));
    }
    
    // Run inference with all inputs
    let inputs_map = input_tensors.into_iter().collect::<std::collections::HashMap<_, _>>();
    let outputs = session.run(inputs_map)?;
    
    // Extract all outputs
    let mut result_outputs = Vec::new();
    for output_name in output_names {
        let output_tensor = outputs[output_name.as_str()].try_extract_array::<f32>()?;
        result_outputs.push(output_tensor.to_owned().into_raw_vec_and_offset().0);
    }
    
    Ok(result_outputs)
}

fn run_onnx_inference_bool(session: &mut Session, input_data_list: &[Vec<bool>], input_infos: &[libinfer::TensorInfo]) -> Result<Vec<Vec<f32>>> {
    // Get output names before mutable borrow
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
    
    // Create input tensors for all inputs (bool data type)
    let mut input_tensors = Vec::new();
    for (i, input_data) in input_data_list.iter().enumerate() {
        let input_info = &input_infos[i];
        // Create tensor with the shape provided from TensorRT (add batch dimension)
        let shape_with_batch: Vec<usize> = std::iter::once(1usize) // batch size 1
            .chain(input_info.dims.iter().map(|&d| d as usize))
            .collect();
        
        let input_tensor = Value::from_array((shape_with_batch, input_data.clone()))?;
        input_tensors.push((input_info.name.as_str(), input_tensor));
    }
    
    // Run inference with all inputs
    let inputs_map = input_tensors.into_iter().collect::<std::collections::HashMap<_, _>>();
    let outputs = session.run(inputs_map)?;
    
    // Extract all outputs
    let mut result_outputs = Vec::new();
    for output_name in output_names {
        let output_tensor = outputs[output_name.as_str()].try_extract_array::<f32>()?;
        result_outputs.push(output_tensor.to_owned().into_raw_vec_and_offset().0);
    }
    
    Ok(result_outputs)
}

fn run_tensorrt_inference(
    engine: &mut UniquePtr<Engine>,
    input_data_list: &[Vec<u8>],
    input_infos: &[libinfer::TensorInfo],
) -> Result<Vec<Vec<f32>>> {
    info!("Info data list len {}", input_data_list.len());
    // Create input tensors for all inputs
    let mut input_tensors = Vec::new();
    for (i, input_data) in input_data_list.iter().enumerate() {
        let input_info = &input_infos[i];
        
        // Debug: Log tensor info
        let expected_size_without_batch: usize = input_info.dims.iter().map(|&d| d as usize).product();
        let _expected_size_with_batch = expected_size_without_batch; // Since we already added batch to data generation
        info!("TensorRT Input {}: name='{}', dims={:?}, data_len={}, expected_without_batch={}", 
              i, input_info.name, input_info.dims, input_data.len(), expected_size_without_batch);
        
        input_tensors.push(InputTensor {
            name: input_info.name.clone(),
            data: input_data.clone(),
            dtype: input_info.dtype.clone(),
        });
    }

    let outputs = engine.pin_mut().infer(&input_tensors)
        .map_err(|e| anyhow!("TensorRT inference failed: {}", e))?;
    
    if outputs.is_empty() {
        return Err(anyhow!("No outputs from TensorRT engine"));
    }

    // Return all outputs
    Ok(outputs.into_iter().map(|o| o.data).collect())
}

fn flat_index_to_multi_dim(flat_index: usize, shape: &[u32]) -> Vec<usize> {
    let mut indices = Vec::with_capacity(shape.len());
    let mut remaining = flat_index;
    
    // Convert shape to usize for calculations
    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    
    // Calculate strides for each dimension
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1;
    for &dim in shape_usize.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }
    strides.reverse();
    
    // Calculate indices for each dimension
    for &stride in &strides {
        indices.push(remaining / stride);
        remaining %= stride;
    }
    
    indices
}

#[derive(Debug, Clone)]
struct OutputDifference {
    max_absolute_diff: f32,
    mean_absolute_diff: f32,
    max_relative_diff: f32,
    mean_relative_diff: f32,
    num_mismatches: usize,
    total_elements: usize,
    mismatch_percentage: f32,
}

fn calculate_output_differences(onnx_output: &[f32], tensorrt_output: &[f32], tolerance: f32, output_shape: &[u32]) -> Result<OutputDifference> {
    if onnx_output.len() != tensorrt_output.len() {
        return Err(anyhow!(
            "Output lengths differ: ONNX={}, TensorRT={}",
            onnx_output.len(),
            tensorrt_output.len()
        ));
    }

    let mut max_absolute_diff = 0.0f32;
    let mut max_relative_diff = 0.0f32;
    let mut total_absolute_diff = 0.0f32;
    let mut total_relative_diff = 0.0f32;
    let mut num_mismatches = 0;

    for (i, (&onnx_val, &trt_val)) in onnx_output.iter().zip(tensorrt_output.iter()).enumerate() {
        let absolute_diff = (onnx_val - trt_val).abs();
        max_absolute_diff = max_absolute_diff.max(absolute_diff);
        total_absolute_diff += absolute_diff;

        let relative_diff = if onnx_val.abs() > 1e-8 {
            absolute_diff / onnx_val.abs()
        } else {
            absolute_diff
        };
        
        max_relative_diff = max_relative_diff.max(relative_diff);
        total_relative_diff += relative_diff;

        if relative_diff > tolerance {
            num_mismatches += 1;
            let multi_dim_indices = flat_index_to_multi_dim(i, output_shape);
            debug!(
                "Mismatch at index {} {:?}: ONNX={:.6}, TensorRT={:.6}, abs_diff={:.6}, rel_diff={:.6}",
                i, multi_dim_indices, onnx_val, trt_val, absolute_diff, relative_diff
            );
        }
    }

    let total_elements = onnx_output.len();
    let mismatch_percentage = 100.0 * num_mismatches as f32 / total_elements as f32;
    
    Ok(OutputDifference {
        max_absolute_diff,
        mean_absolute_diff: total_absolute_diff / total_elements as f32,
        max_relative_diff,
        mean_relative_diff: total_relative_diff / total_elements as f32,
        num_mismatches,
        total_elements,
        mismatch_percentage,
    })
}

#[derive(Debug, Clone)]
struct BenchmarkResults {
    onnx_latencies: Vec<Duration>,
    tensorrt_latencies: Vec<Duration>,
    onnx_avg_latency: f32,
    tensorrt_avg_latency: f32,
    onnx_throughput: f32,
    tensorrt_throughput: f32,
    speedup_factor: f32,
}

fn benchmark_inference(
    session: &mut Session,
    engine: &mut UniquePtr<Engine>,
    input_data_list_u8: &[Vec<u8>],
    input_data_list_f32: &Option<Vec<Vec<f32>>>,
    input_dims: &[libinfer::TensorInfo],
    input_data_type: TensorDataType,
    num_runs: usize,
) -> Result<BenchmarkResults> {
    info!("Warming up inference paths (10 iterations)...");
    
    // Warmup - run a few iterations to stabilize performance
    for i in 0..10 {
        if i % 5 == 0 {
            info!("  Warmup progress: {}/10", i + 1);
        }
        match input_data_type {
            TensorDataType::UINT8 => {
                let _ = run_onnx_inference_u8(session, input_data_list_u8, input_dims)?;
                let _ = run_tensorrt_inference(engine, input_data_list_u8, input_dims)?;
            }
            TensorDataType::FP32 => {
                if let Some(ref input_f32) = input_data_list_f32 {
                    let _ = run_onnx_inference_f32(session, input_f32, input_dims)?;
                    let _ = run_tensorrt_inference(engine, input_data_list_u8, input_dims)?;
                }
            }
            TensorDataType::INT64 | TensorDataType::BOOL => {
                // For INT64 and BOOL, we skip warmup as they require additional data preparation
                // that's handled in the main benchmark loop
                warn!("Skipping warmup for {:?} data type", input_data_type);
            }
        }
    }

    info!("Running {} benchmark iterations...", num_runs);
    
    // Benchmark ONNX with progress reporting
    info!("Benchmarking ONNX Runtime...");
    let benchmark_start = Instant::now();
    let mut recent_latencies = Vec::new();
    
    let onnx_latencies: Vec<Duration> = (0..num_runs)
        .map(|i| {
            let start = Instant::now();
            let _ = match input_data_type {
                TensorDataType::UINT8 => run_onnx_inference_u8(session, input_data_list_u8, input_dims),
                TensorDataType::FP32 => {
                    if let Some(ref input_f32) = input_data_list_f32 {
                        run_onnx_inference_f32(session, input_f32, input_dims)
                    } else {
                        Err(anyhow!("Missing f32 input data"))
                    }
                }
                TensorDataType::INT64 | TensorDataType::BOOL => {
                    Err(anyhow!("Benchmarking for {:?} data type requires additional data preparation - not yet implemented", input_data_type))
                }
            };
            let iteration_time = start.elapsed();
            recent_latencies.push(iteration_time);
            
            // Report progress every 10 iterations (but not on the first)
            if i > 0 && (i + 1) % 10 == 0 {
                let elapsed = benchmark_start.elapsed().as_secs_f32();
                let estimated_total = elapsed * num_runs as f32 / (i + 1) as f32;
                let remaining = estimated_total - elapsed;
                
                // Calculate average latency for last 10 iterations
                let recent_avg = if recent_latencies.len() >= 10 {
                    let last_10: Vec<Duration> = recent_latencies.iter().rev().take(10).cloned().collect();
                    last_10.iter().map(|t| t.as_secs_f32()).sum::<f32>() / 10.0
                } else {
                    recent_latencies.iter().map(|t| t.as_secs_f32()).sum::<f32>() / recent_latencies.len() as f32
                };
                
                info!("  ONNX progress: {}/{} ({:.1}%) - Avg latency: {:.4}s - ETA: {:.1}s", 
                      i + 1, num_runs, 100.0 * (i + 1) as f32 / num_runs as f32, recent_avg, remaining);
            }
            
            iteration_time
        })
        .collect();

    // Benchmark TensorRT with progress reporting
    info!("Benchmarking TensorRT...");
    let tensorrt_start = Instant::now();
    let mut recent_trt_latencies = Vec::new();
    
    let tensorrt_latencies: Vec<Duration> = (0..num_runs)
        .map(|i| {
            let start = Instant::now();
            let _ = run_tensorrt_inference(engine, input_data_list_u8, input_dims);
            let iteration_time = start.elapsed();
            recent_trt_latencies.push(iteration_time);
            
            // Report progress every 10 iterations (but not on the first)
            if i > 0 && (i + 1) % 10 == 0 {
                let elapsed = tensorrt_start.elapsed().as_secs_f32();
                let estimated_total = elapsed * num_runs as f32 / (i + 1) as f32;
                let remaining = estimated_total - elapsed;
                
                // Calculate average latency for last 10 iterations
                let recent_avg = if recent_trt_latencies.len() >= 10 {
                    let last_10: Vec<Duration> = recent_trt_latencies.iter().rev().take(10).cloned().collect();
                    last_10.iter().map(|t| t.as_secs_f32()).sum::<f32>() / 10.0
                } else {
                    recent_trt_latencies.iter().map(|t| t.as_secs_f32()).sum::<f32>() / recent_trt_latencies.len() as f32
                };
                
                info!("  TensorRT progress: {}/{} ({:.1}%) - Avg latency: {:.4}s - ETA: {:.1}s", 
                      i + 1, num_runs, 100.0 * (i + 1) as f32 / num_runs as f32, recent_avg, remaining);
            }
            
            iteration_time
        })
        .collect();

    // Calculate statistics
    let onnx_avg_latency = onnx_latencies.iter().map(|t| t.as_secs_f32()).sum::<f32>() / num_runs as f32;
    let tensorrt_avg_latency = tensorrt_latencies.iter().map(|t| t.as_secs_f32()).sum::<f32>() / num_runs as f32;
    
    let onnx_throughput = 1.0 / onnx_avg_latency;
    let tensorrt_throughput = 1.0 / tensorrt_avg_latency;
    let speedup_factor = onnx_avg_latency / tensorrt_avg_latency;

    Ok(BenchmarkResults {
        onnx_latencies,
        tensorrt_latencies,
        onnx_avg_latency,
        tensorrt_avg_latency,
        onnx_throughput,
        tensorrt_throughput,
        speedup_factor,
    })
}

fn main() -> Result<()> {
    // Initialize tracing subscriber for logging
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");

    let args = Args::parse();
    
    // Storage for collecting all results
    let mut all_output_differences: Vec<Vec<OutputDifference>> = Vec::new();
    let mut benchmark_results: Option<BenchmarkResults> = None;

    // Initialize seeded RNG for reproducible results
    let mut rng = StdRng::seed_from_u64(args.seed);
    info!("Using random seed: {}", args.seed);

    info!("Loading ONNX model: {}", args.onnx.display());
    
    let mut session = match args.onnx_ep {
        ExecutionProvider::CPU => {
            Session::builder()?
                .with_intra_threads(4)?
                .commit_from_file(&args.onnx)?
        },
        ExecutionProvider::CUDA => {
            Session::builder()?
                .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(&args.onnx)?
        },
        ExecutionProvider::TensorRT => {
            Session::builder()?
                .with_execution_providers([TensorRTExecutionProvider::default().build().error_on_failure()])?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .commit_from_file(&args.onnx)?
        }
    };

    info!("ONNX model loaded successfully");

    // Load TensorRT engine
    info!("Loading TensorRT engine: {}", args.engine.display());
    let options = Options {
        path: args.engine.to_string_lossy().to_string(),
        device_index: args.device,
    };

    let mut engine = Engine::new(&options)
        .map_err(|e| anyhow!("Failed to load TensorRT engine: {}", e))?;

    info!("TensorRT engine loaded successfully");

    // Get model information
    let input_dims = engine.get_input_dims();
    let output_dims = engine.get_output_dims();
    
    // Check if all input tensors have the same data type (for backward compatibility)
    let input_data_type = if input_dims.is_empty() {
        return Err(anyhow!("No input tensors found in TensorRT engine"));
    } else if input_dims.iter().all(|tensor| tensor.dtype == input_dims[0].dtype) {
        input_dims[0].dtype.clone()
    } else {
        return Err(anyhow!(
            "Mixed input data types are not yet supported in comparison mode. Found types: {:?}",
            input_dims.iter().map(|t| (&t.name, &t.dtype)).collect::<Vec<_>>()
        ));
    };
    
    if output_dims.is_empty() {
        return Err(anyhow!("No output tensors found in TensorRT engine"));
    }

    // Validate input/output count matches between ONNX and TensorRT
    if session.inputs.len() != input_dims.len() {
        return Err(anyhow!(
            "Input count mismatch: ONNX has {} inputs, TensorRT has {} inputs",
            session.inputs.len(), input_dims.len()
        ));
    }
    
    if session.outputs.len() != output_dims.len() {
        return Err(anyhow!(
            "Output count mismatch: ONNX has {} outputs, TensorRT has {} outputs", 
            session.outputs.len(), output_dims.len()
        ));
    }

    // Validate input names match (order-independent check)
    let onnx_input_names: std::collections::HashSet<String> = session.inputs.iter().map(|i| i.name.clone()).collect();
    let trt_input_names: std::collections::HashSet<String> = input_dims.iter().map(|i| i.name.clone()).collect();
    
    if onnx_input_names != trt_input_names {
        warn!("Input name mismatch detected:");
        warn!("ONNX inputs: {:?}", onnx_input_names);
        warn!("TensorRT inputs: {:?}", trt_input_names);
        warn!("This may cause incorrect tensor mapping");
    }

    // Validate output names match (order-independent check)  
    let onnx_output_names: std::collections::HashSet<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
    let trt_output_names: std::collections::HashSet<String> = output_dims.iter().map(|o| o.name.clone()).collect();
    
    if onnx_output_names != trt_output_names {
        warn!("Output name mismatch detected:");
        warn!("ONNX outputs: {:?}", onnx_output_names);
        warn!("TensorRT outputs: {:?}", trt_output_names);
        warn!("This may cause incorrect comparison");
    }

    info!("Model validation passed:");
    info!("- {} input tensors", input_dims.len());
    info!("- {} output tensors", output_dims.len());
    info!("- Input data type: {:?}", input_data_type);

    info!("Detailed tensor information:");
    
    for (i, input_info) in input_dims.iter().enumerate() {
        let tensor_size: usize = input_info.dims.iter().map(|&d| d as usize).product();
        info!("TensorRT input {}: '{}' dims={:?} size={}", i, input_info.name, input_info.dims, tensor_size);
    }
    
    for (i, output_info) in output_dims.iter().enumerate() {
        let tensor_size: usize = output_info.dims.iter().map(|&d| d as usize).product();
        info!("TensorRT output {}: '{}' dims={:?} size={}", i, output_info.name, output_info.dims, tensor_size);
    }

    for (i, onnx_input) in session.inputs.iter().enumerate() {
        info!("ONNX input {}: '{}'", i, onnx_input.name);
    }
    
    for (i, onnx_output) in session.outputs.iter().enumerate() {
        info!("ONNX output {}: '{}'", i, onnx_output.name);
    }

    // Generate a single set of random inputs for benchmarking and comparison
    info!("Generating random input data...");
    let mut input_data_list_u8 = Vec::new();
    let mut input_data_list_f32: Option<Vec<Vec<f32>>> = None;
    
    for input_info in &input_dims {
        // Create dimensions with batch size = 1
        let dims_with_batch: Vec<u32> = std::iter::once(1u32)
            .chain(input_info.dims.iter().cloned())
            .collect();
        
        let data_u8 = match input_data_type {
            TensorDataType::UINT8 => {
                generate_random_input_u8(&dims_with_batch, &mut rng)
            }
            TensorDataType::FP32 => {
                let data_f32 = generate_random_input_f32(&dims_with_batch, &mut rng);
                f32_to_u8(&data_f32)
            }
            TensorDataType::INT64 => {
                let data_int64 = generate_random_input_int64(&dims_with_batch, &mut rng);
                int64_to_u8(&data_int64)
            }
            TensorDataType::BOOL => {
                let data_bool = generate_random_input_bool(&dims_with_batch, &mut rng);
                bool_to_u8(&data_bool)
            }
        };
        input_data_list_u8.push(data_u8);
    }

    // Prepare specific data types for ONNX inference
    let mut input_data_list_int64: Option<Vec<Vec<i64>>> = None;
    let mut input_data_list_bool: Option<Vec<Vec<bool>>> = None;
    
    match input_data_type {
        TensorDataType::FP32 => {
            let mut f32_inputs = Vec::new();
            for (tensor_idx, u8_data) in input_data_list_u8.iter().enumerate() {
                if u8_data.len() % 4 != 0 {
                    return Err(anyhow!(
                        "Invalid u8 data length for tensor {}: {} bytes (not divisible by 4)",
                        tensor_idx, u8_data.len()
                    ));
                }
                
                let f32_data: Vec<f32> = u8_data
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect();
                
                f32_inputs.push(f32_data);
            }
            input_data_list_f32 = Some(f32_inputs);
        }
        TensorDataType::INT64 => {
            let mut int64_inputs = Vec::new();
            for (tensor_idx, u8_data) in input_data_list_u8.iter().enumerate() {
                if u8_data.len() % 8 != 0 {
                    return Err(anyhow!(
                        "Invalid u8 data length for tensor {}: {} bytes (not divisible by 8)",
                        tensor_idx, u8_data.len()
                    ));
                }
                
                let int64_data: Vec<i64> = u8_data
                    .chunks_exact(8)
                    .map(|bytes| i64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7]
                    ]))
                    .collect();
                
                int64_inputs.push(int64_data);
            }
            input_data_list_int64 = Some(int64_inputs);
        }
        TensorDataType::BOOL => {
            let mut bool_inputs = Vec::new();
            for u8_data in input_data_list_u8.iter() {
                let bool_data: Vec<bool> = u8_data
                    .iter()
                    .map(|&byte| byte != 0)
                    .collect();
                
                bool_inputs.push(bool_data);
            }
            input_data_list_bool = Some(bool_inputs);
        }
        TensorDataType::UINT8 => {
            // No additional preparation needed for UINT8
        }
    }

    // Run performance benchmarking if not skipped
    if !args.skip_benchmark {
        info!("Running performance benchmark...");
        benchmark_results = Some(benchmark_inference(
            &mut session,
            &mut engine,
            &input_data_list_u8,
            &input_data_list_f32,
            &input_dims,
            input_data_type,
            args.benchmark_iterations,
        )?);
    }

    // Run correctness comparison tests
    info!("Running {} correctness comparison iterations...", args.iterations);
    let correctness_start = Instant::now();
    
    for iteration in 1..=args.iterations {
        // Progress reporting for correctness iterations
        if args.iterations >= 20 && iteration % (args.iterations / 10).max(1) == 0 {
            let elapsed = correctness_start.elapsed().as_secs_f32();
            let estimated_total = elapsed * args.iterations as f32 / iteration as f32;
            let remaining = estimated_total - elapsed;
            info!("Correctness progress: {}/{} ({:.1}%) - ETA: {:.1}s", 
                  iteration, args.iterations, 100.0 * iteration as f32 / args.iterations as f32, remaining);
        } else if args.iterations < 20 {
            info!("Correctness iteration {}/{}", iteration, args.iterations);
        }

        // Run ONNX and TensorRT inference based on data type
        let (onnx_outputs, tensorrt_outputs) = match input_data_type {
            TensorDataType::UINT8 => {
                let onnx_outputs = run_onnx_inference_u8(&mut session, &input_data_list_u8, &input_dims)
                    .map_err(|e| anyhow!("ONNX inference failed: {}", e))?;
                let tensorrt_outputs = run_tensorrt_inference(&mut engine, &input_data_list_u8, &input_dims)
                    .map_err(|e| anyhow!("TensorRT inference failed: {}", e))?;
                (onnx_outputs, tensorrt_outputs)
            }
            TensorDataType::FP32 => {
                if let Some(ref input_f32) = input_data_list_f32 {
                    let onnx_outputs = run_onnx_inference_f32(&mut session, input_f32, &input_dims)
                        .map_err(|e| anyhow!("ONNX inference failed: {}", e))?;
                    let tensorrt_outputs = run_tensorrt_inference(&mut engine, &input_data_list_u8, &input_dims)
                        .map_err(|e| anyhow!("TensorRT inference failed: {}", e))?;
                    (onnx_outputs, tensorrt_outputs)
                } else {
                    return Err(anyhow!("Missing f32 input data for FP32 model"));
                }
            }
            TensorDataType::INT64 => {
                if let Some(ref input_int64) = input_data_list_int64 {
                    let onnx_outputs = run_onnx_inference_int64(&mut session, input_int64, &input_dims)
                        .map_err(|e| anyhow!("ONNX inference failed: {}", e))?;
                    let tensorrt_outputs = run_tensorrt_inference(&mut engine, &input_data_list_u8, &input_dims)
                        .map_err(|e| anyhow!("TensorRT inference failed: {}", e))?;
                    (onnx_outputs, tensorrt_outputs)
                } else {
                    return Err(anyhow!("Missing int64 input data for INT64 model"));
                }
            }
            TensorDataType::BOOL => {
                if let Some(ref input_bool) = input_data_list_bool {
                    let onnx_outputs = run_onnx_inference_bool(&mut session, input_bool, &input_dims)
                        .map_err(|e| anyhow!("ONNX inference failed: {}", e))?;
                    let tensorrt_outputs = run_tensorrt_inference(&mut engine, &input_data_list_u8, &input_dims)
                        .map_err(|e| anyhow!("TensorRT inference failed: {}", e))?;
                    (onnx_outputs, tensorrt_outputs)
                } else {
                    return Err(anyhow!("Missing bool input data for BOOL model"));
                }
            }
        };

        // Compare all outputs and collect differences (don't exit on errors)
        if onnx_outputs.len() != tensorrt_outputs.len() {
            warn!(
                "Output count mismatch during inference iteration {}: ONNX returned {} outputs, TensorRT returned {} outputs",
                iteration, onnx_outputs.len(), tensorrt_outputs.len()
            );
            continue;
        }
        
        let mut iteration_differences = Vec::new();
        for (i, (onnx_output, tensorrt_output)) in onnx_outputs.iter().zip(tensorrt_outputs.iter()).enumerate() {
            // Get output shape (with batch dimension)
            let output_shape_with_batch: Vec<u32> = std::iter::once(1u32) // batch = 1
                .chain(output_dims[i].dims.iter().cloned())
                .collect();
            
            match calculate_output_differences(onnx_output, tensorrt_output, args.tolerance, &output_shape_with_batch) {
                Ok(diff) => {
                    debug!("Output {} iteration {}: max_abs_diff={:.6}, mean_abs_diff={:.6}, mismatches={}/{} ({:.2}%)",
                           i, iteration, diff.max_absolute_diff, diff.mean_absolute_diff, 
                           diff.num_mismatches, diff.total_elements, diff.mismatch_percentage);
                    iteration_differences.push(diff);
                }
                Err(e) => {
                    warn!("Failed to compare output {} in iteration {}: {}", i, iteration, e);
                    // Create a placeholder difference indicating failure
                    iteration_differences.push(OutputDifference {
                        max_absolute_diff: f32::INFINITY,
                        mean_absolute_diff: f32::INFINITY,
                        max_relative_diff: f32::INFINITY,
                        mean_relative_diff: f32::INFINITY,
                        num_mismatches: usize::MAX,
                        total_elements: 0,
                        mismatch_percentage: 100.0,
                    });
                }
            }
        }
        all_output_differences.push(iteration_differences);
    }

    // Print comprehensive results
    print_comprehensive_results(&all_output_differences, &benchmark_results, &args);

    Ok(())
}

fn print_comprehensive_results(
    all_differences: &[Vec<OutputDifference>], 
    benchmark_results: &Option<BenchmarkResults>,
    args: &Args
) {
    info!("\n{}", "=".repeat(80));
    info!("COMPREHENSIVE COMPARISON RESULTS");
    info!("{}", "=".repeat(80));
    
    // Print model information
    info!("Model Comparison:");
    info!("  ONNX Model: {}", args.onnx.display());
    info!("  TensorRT Engine: {}", args.engine.display());
    info!("  Tolerance: {:.6}", args.tolerance);
    info!("  Correctness Iterations: {}", args.iterations);
    
    // Print performance results if available
    if let Some(bench) = benchmark_results {
        info!("\nPERFORMANCE RESULTS:");
        info!("  Benchmark Iterations: {}", bench.onnx_latencies.len());
        info!("  ONNX Average Latency: {:.6} sec", bench.onnx_avg_latency);
        info!("  TensorRT Average Latency: {:.6} sec", bench.tensorrt_avg_latency);
        info!("  ONNX Throughput: {:.2} inferences/sec", bench.onnx_throughput);
        info!("  TensorRT Throughput: {:.2} inferences/sec", bench.tensorrt_throughput);
        info!("  TensorRT Speedup: {:.2}x faster than ONNX", bench.speedup_factor);
        
        // Calculate percentiles for latencies
        let mut onnx_sorted = bench.onnx_latencies.clone();
        let mut trt_sorted = bench.tensorrt_latencies.clone();
        onnx_sorted.sort();
        trt_sorted.sort();
        
        let onnx_p50 = onnx_sorted[onnx_sorted.len() / 2].as_secs_f32();
        let onnx_p95 = onnx_sorted[onnx_sorted.len() * 95 / 100].as_secs_f32();
        let onnx_p99 = onnx_sorted[onnx_sorted.len() * 99 / 100].as_secs_f32();
        
        let trt_p50 = trt_sorted[trt_sorted.len() / 2].as_secs_f32();
        let trt_p95 = trt_sorted[trt_sorted.len() * 95 / 100].as_secs_f32();
        let trt_p99 = trt_sorted[trt_sorted.len() * 99 / 100].as_secs_f32();
        
        info!("\nLATENCY PERCENTILES:");
        info!("  ONNX    - P50: {:.6}s, P95: {:.6}s, P99: {:.6}s", onnx_p50, onnx_p95, onnx_p99);
        info!("  TensorRT - P50: {:.6}s, P95: {:.6}s, P99: {:.6}s", trt_p50, trt_p95, trt_p99);
    } else {
        info!("\nPERFORMANCE RESULTS: Skipped (--skip-benchmark was specified)");
    }
    
    // Print correctness results
    info!("\nCORRECTNESS RESULTS:");
    
    if all_differences.is_empty() {
        info!("  No correctness iterations completed.");
        return;
    }
    
    let num_outputs = all_differences[0].len();
    info!("  Number of outputs: {}", num_outputs);
    
    for output_idx in 0..num_outputs {
        info!("\n  Output {} Analysis:", output_idx);
        
        // Collect stats across all iterations for this output
        let output_diffs: Vec<&OutputDifference> = all_differences
            .iter()
            .map(|iter_diffs| &iter_diffs[output_idx])
            .collect();
        
        // Skip if all iterations had failures
        let valid_diffs: Vec<&OutputDifference> = output_diffs
            .iter()
            .filter(|diff| diff.max_absolute_diff.is_finite())
            .cloned()
            .collect();
            
        if valid_diffs.is_empty() {
            info!("    All iterations failed comparison for this output");
            continue;
        }
        
        // Calculate aggregate statistics
        let max_abs_diff = valid_diffs.iter().map(|d| d.max_absolute_diff).fold(0.0f32, f32::max);
        let avg_max_abs_diff = valid_diffs.iter().map(|d| d.max_absolute_diff).sum::<f32>() / valid_diffs.len() as f32;
        let avg_mean_abs_diff = valid_diffs.iter().map(|d| d.mean_absolute_diff).sum::<f32>() / valid_diffs.len() as f32;
        
        let max_rel_diff = valid_diffs.iter().map(|d| d.max_relative_diff).fold(0.0f32, f32::max);
        let avg_max_rel_diff = valid_diffs.iter().map(|d| d.max_relative_diff).sum::<f32>() / valid_diffs.len() as f32;
        let avg_mean_rel_diff = valid_diffs.iter().map(|d| d.mean_relative_diff).sum::<f32>() / valid_diffs.len() as f32;
        
        let avg_mismatch_pct = valid_diffs.iter().map(|d| d.mismatch_percentage).sum::<f32>() / valid_diffs.len() as f32;
        let max_mismatch_pct = valid_diffs.iter().map(|d| d.mismatch_percentage).fold(0.0f32, f32::max);
        
        let sample_total_elements = valid_diffs[0].total_elements;
        
        info!("    Total Elements: {}", sample_total_elements);
        info!("    Valid Iterations: {}/{}", valid_diffs.len(), all_differences.len());
        info!("    Absolute Differences:");
        info!("      Maximum across all iterations: {:.6}", max_abs_diff);
        info!("      Average maximum per iteration: {:.6}", avg_max_abs_diff);
        info!("      Average mean per iteration: {:.6}", avg_mean_abs_diff);
        info!("    Relative Differences:");
        info!("      Maximum across all iterations: {:.6}", max_rel_diff);
        info!("      Average maximum per iteration: {:.6}", avg_max_rel_diff);
        info!("      Average mean per iteration: {:.6}", avg_mean_rel_diff);
        info!("    Tolerance Violations:");
        info!("      Average mismatch percentage: {:.2}%", avg_mismatch_pct);
        info!("      Maximum mismatch percentage: {:.2}%", max_mismatch_pct);
        
        // Determine overall status
        if max_mismatch_pct > 1.0 {
            info!("    Status: ⚠️  SIGNIFICANT DIFFERENCES (>{:.1}% mismatch)", 1.0);
        } else if avg_mismatch_pct > 0.1 {
            info!("    Status: ⚠️  MINOR DIFFERENCES (>{:.1}% avg mismatch)", 0.1);
        } else {
            info!("    Status: ✅ OUTPUTS MATCH WITHIN TOLERANCE");
        }
    }
    
    info!("\n{}", "=".repeat(80));
    
    // Summary recommendation
    let has_significant_diffs = (0..num_outputs).any(|output_idx| {
        let valid_diffs: Vec<&OutputDifference> = all_differences
            .iter()
            .map(|iter_diffs| &iter_diffs[output_idx])
            .filter(|diff| diff.max_absolute_diff.is_finite())
            .collect();
        
        !valid_diffs.is_empty() && 
        valid_diffs.iter().any(|d| d.mismatch_percentage > 1.0)
    });
    
    if has_significant_diffs {
        info!("OVERALL STATUS: ⚠️  Models have significant output differences");
        info!("RECOMMENDATION: Review model conversion or increase tolerance");
    } else {
        info!("OVERALL STATUS: ✅ Models produce consistent results");
        if let Some(bench) = benchmark_results {
            info!("RECOMMENDATION: TensorRT provides {:.1}x speedup with good accuracy", bench.speedup_factor);
        }
    }
}
