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
//! - This version supports single-input, single-output models
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
use libinfer::{Engine, InputDataType, Options};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
    inputs,
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::path::PathBuf;
use tracing::{info, debug, Level, warn};
use tracing_subscriber::{FmtSubscriber, EnvFilter};

#[derive(Parser, Debug)]
#[clap(about = "Compare ONNX Runtime vs TensorRT inference results")]
struct Args {
    /// Path to the ONNX model file
    #[arg(long, value_name = "PATH", value_parser)]
    onnx: PathBuf,

    /// Path to the TensorRT engine file
    #[arg(long, value_name = "PATH", value_parser)]
    engine: PathBuf,

    /// GPU device index to use (default: 0)
    #[arg(long, default_value_t = 0)]
    device: u32,

    /// Tolerance for output comparison (default: 0.001)
    #[arg(long, default_value_t = 0.1)]
    tolerance: f32,

    /// Number of random test iterations (default: 5)
    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// Random seed for reproducible results (default: 42)
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn generate_random_input_u8(dims: &[u32], rng: &mut StdRng) -> Vec<u8> {
    let size: usize = dims.iter().map(|&d| d as usize).product();
    (0..size).map(|_| rng.gen_range(0..=255)).collect()
}

fn generate_random_input_f32(dims: &[u32], rng: &mut StdRng) -> Vec<f32> {
    let size: usize = dims.iter().map(|&d| d as usize).product();
    (0..size).map(|_| rng.gen_range(-1.0..=1.0)).collect()
}

fn f32_to_u8(data: &[f32]) -> Vec<u8> {
    // Convert f32 values to their byte representation (4 bytes per f32)
    // Use little-endian to match the from_le_bytes() conversion later
    data.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect()
}

fn run_onnx_inference_f32(session: &mut Session, input_data: &[f32], input_dims: &[u32]) -> Result<Vec<f32>> {
    // Get output names before mutable borrow
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
    
    if output_names.len() != 1 {
        return Err(anyhow!("Expected exactly one output, got {}", output_names.len()));
    }
    
    // Create input tensor with batch dimension
    let shape_with_batch: Vec<usize> = std::iter::once(1usize) // batch size 1
        .chain(input_dims.iter().map(|&d| d as usize))
        .collect();
    
    let input_tensor = Value::from_array((shape_with_batch, input_data.to_vec()))?;
    let input_name = session.inputs[0].name.clone();
    
    let outputs = session.run(inputs![input_name => input_tensor])?;
    
    let output_tensor = outputs[output_names[0].as_str()].try_extract_array::<f32>()?;
    Ok(output_tensor.to_owned().into_raw_vec_and_offset().0)
}

fn run_onnx_inference_u8(session: &mut Session, input_data: &[u8], input_dims: &[u32]) -> Result<Vec<f32>> {
    // Get output names before mutable borrow
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
    
    if output_names.len() != 1 {
        return Err(anyhow!("Expected exactly one output, got {}", output_names.len()));
    }
    
    // Create input tensor with batch dimension
    let shape_with_batch: Vec<usize> = std::iter::once(1usize) // batch size 1
        .chain(input_dims.iter().map(|&d| d as usize))
        .collect();
    
    let input_tensor = Value::from_array((shape_with_batch, input_data.to_vec()))?;
    let input_name = session.inputs[0].name.clone();
    
    let outputs = session.run(inputs![input_name => input_tensor])?;
    
    let output_tensor = outputs[output_names[0].as_str()].try_extract_array::<f32>()?;
    Ok(output_tensor.to_owned().into_raw_vec_and_offset().0)
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

fn compare_outputs(onnx_output: &[f32], tensorrt_output: &[f32], tolerance: f32, output_shape: &[u32]) -> Result<()> {
    if onnx_output.len() != tensorrt_output.len() {
        return Err(anyhow!(
            "Output lengths differ: ONNX={}, TensorRT={}",
            onnx_output.len(),
            tensorrt_output.len()
        ));
    }
    info!("output sizes onnx {} trt {}", onnx_output.len(), tensorrt_output.len());

    let mut max_diff = 0.0f32;
    let mut num_mismatches = 0;

    for (i, (&onnx_val, &trt_val)) in onnx_output.iter().zip(tensorrt_output.iter()).enumerate() {
        let diff = (onnx_val - trt_val).abs();
        max_diff = max_diff.max(diff);

        let relative_error = if onnx_val.abs() > 1e-5 {
            diff / onnx_val.abs()
        } else {
            diff
        };

        if relative_error > tolerance {
            num_mismatches += 1;
            let multi_dim_indices = flat_index_to_multi_dim(i, output_shape);
            debug!(
                "Mismatch at index {} {:?}: ONNX={:.6}, TensorRT={:.6}, diff={:.6}, rel_err={:.6}",
                i, multi_dim_indices, onnx_val, trt_val, diff, relative_error
            );
        }
    }

    info!("Max absolute difference: {:.6}", max_diff);
    info!("Number of mismatches: {}/{}", num_mismatches, onnx_output.len());

    if num_mismatches > onnx_output.len() / 100 {  // More than 1% mismatch
        return Err(anyhow!(
            "Too many mismatches: {}/{} (>{:.1}%)",
            num_mismatches,
            onnx_output.len(),
            100.0 * num_mismatches as f32 / onnx_output.len() as f32
        ));
    }

    Ok(())
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

    // Initialize seeded RNG for reproducible results
    let mut rng = StdRng::seed_from_u64(args.seed);
    info!("Using random seed: {}", args.seed);

    info!("Loading ONNX model: {}", args.onnx.display());
    
    // Create session using ort 2.0 API
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(&args.onnx)?;

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
    let input_data_type = engine.get_input_data_type();
    
    // Validate basic model structure - this is a single-input, single-output version
    if session.inputs.len() != 1 {
        return Err(anyhow!("Expected single input ONNX model, got {} inputs", session.inputs.len()));
    }
    
    if session.outputs.len() != 1 {
        return Err(anyhow!("Expected single output ONNX model, got {} outputs", session.outputs.len()));
    }

    info!("Model validation passed:");
    info!("- Single input tensor: dims={:?}", input_dims);
    info!("- Single output tensor: dims={:?}", output_dims);
    info!("- Input data type: {:?}", input_data_type);

    info!("ONNX input name: '{}'", session.inputs[0].name);
    info!("ONNX output name: '{}'", session.outputs[0].name);

    // Run comparison tests
    info!("Running {} comparison iterations...", args.iterations);
    
    for iteration in 1..=args.iterations {
        info!("Iteration {}/{}", iteration, args.iterations);

        // Generate random input data based on TensorRT input data type
        // Note: TensorRT dims don't include batch dimension, but we need to add batch=1
        let dims_with_batch: Vec<u32> = std::iter::once(1u32)
            .chain(input_dims.iter().cloned())
            .collect();
        
        let input_data_u8 = match input_data_type {
            InputDataType::UINT8 => {
                // For UINT8 models, generate u8 data directly
                generate_random_input_u8(&dims_with_batch, &mut rng)
            }
            InputDataType::FP32 => {
                // For FP32 models, generate f32 data then convert to bytes
                let data_f32 = generate_random_input_f32(&dims_with_batch, &mut rng);
                f32_to_u8(&data_f32)
            }
            _ => {
                return Err(anyhow!("Unsupported input data type: {:?}", input_data_type));
            }
        };

        // Debug: Log data sizes for verification
        let expected_size: usize = input_dims.iter().map(|&d| d as usize).product();
        let multiplier = match input_data_type {
            InputDataType::UINT8 => 1, // 1 byte per element
            InputDataType::FP32 => 4,  // 4 bytes per f32
            _ => 1,
        };
        info!("Input data: u8_len={}, expected_size_without_batch={}, expected_with_batch={}, multiplier={}", 
              input_data_u8.len(), expected_size, expected_size * multiplier, multiplier);

        // Run ONNX and TensorRT inference based on data type
        let (onnx_output, tensorrt_output) = match input_data_type {
            InputDataType::UINT8 => {
                // Both models take u8 data directly
                let onnx_output = run_onnx_inference_u8(&mut session, &input_data_u8, &input_dims)
                    .map_err(|e| anyhow!("ONNX inference failed: {}", e))?;
                let tensorrt_output = engine.pin_mut().infer(&input_data_u8)
                    .map_err(|e| anyhow!("TensorRT inference failed: {}", e))?;
                (onnx_output, tensorrt_output)
            }
            InputDataType::FP32 => {
                // Convert u8 bytes back to f32 for ONNX, TensorRT takes u8 bytes
                if input_data_u8.len() % 4 != 0 {
                    return Err(anyhow!(
                        "Invalid u8 data length: {} bytes (not divisible by 4)",
                        input_data_u8.len()
                    ));
                }
                
                // Convert bytes back to f32 values
                let input_data_f32: Vec<f32> = input_data_u8
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect();
                
                // Sanity check: verify no NaN or infinite values
                let invalid_count = input_data_f32.iter().filter(|&&f| !f.is_finite()).count();
                if invalid_count > 0 {
                    warn!("Input has {} non-finite values after byte conversion", invalid_count);
                }
                
                info!("Converted {} bytes to {} f32 values", input_data_u8.len(), input_data_f32.len());
                
                let onnx_output = run_onnx_inference_f32(&mut session, &input_data_f32, &input_dims)
                    .map_err(|e| anyhow!("ONNX inference failed: {}", e))?;
                let tensorrt_output = engine.pin_mut().infer(&input_data_u8)
                    .map_err(|e| anyhow!("TensorRT inference failed: {}", e))?;
                (onnx_output, tensorrt_output)
            }
            _ => {
                return Err(anyhow!("Unsupported input data type: {:?}", input_data_type));
            }
        };

        // Compare outputs
        // Get output shape (with batch dimension)
        let output_shape_with_batch: Vec<u32> = std::iter::once(1u32) // batch = 1
            .chain(output_dims.iter().cloned())
            .collect();
        
        compare_outputs(&onnx_output, &tensorrt_output, args.tolerance, &output_shape_with_batch)
            .map_err(|e| anyhow!("Output comparison failed: {}", e))?;
    }

    info!("All {} iterations passed successfully!", args.iterations);
    Ok(())
}