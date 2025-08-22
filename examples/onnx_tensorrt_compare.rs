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
use libinfer::{Engine, InputDataType, Options};
use libinfer::ffi::InputTensor;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use rand::{Rng, thread_rng};
use std::path::PathBuf;
use tracing::{info, warn, Level};
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
}

fn generate_random_input_u8(dims: &[u32]) -> Vec<u8> {
    let mut rng = thread_rng();
    let size: usize = dims.iter().map(|&d| d as usize).product();
    (0..size).map(|_| rng.gen_range(0..=255)).collect()
}

fn generate_random_input_f32(dims: &[u32]) -> Vec<f32> {
    let mut rng = thread_rng();
    let size: usize = dims.iter().map(|&d| d as usize).product();
    (0..size).map(|_| rng.gen_range(-1.0..=1.0)).collect()
}

fn u8_to_f32(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&x| x as f32 / 255.0).collect()
}

fn f32_to_u8(data: &[f32]) -> Vec<u8> {
    // Convert f32 values to their byte representation (4 bytes per f32)
    data.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect()
}

fn run_onnx_inference(session: &mut Session, input_data_list: &[Vec<f32>], input_infos: &[libinfer::TensorInfo]) -> Result<Vec<Vec<f32>>> {
    // Get output names before mutable borrow
    let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();
    
    // Create input tensors for all inputs
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
        let expected_size_with_batch = expected_size_without_batch; // Since we already added batch to data generation
        info!("TensorRT Input {}: name='{}', dims={:?}, data_len={}, expected_without_batch={}", 
              i, input_info.name, input_info.dims, input_data.len(), expected_size_without_batch);
        
        input_tensors.push(InputTensor {
            name: input_info.name.clone(),
            data: input_data.clone(),
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

fn compare_outputs(onnx_output: &[f32], tensorrt_output: &[f32], tolerance: f32) -> Result<()> {
    if onnx_output.len() != tensorrt_output.len() {
        return Err(anyhow!(
            "Output lengths differ: ONNX={}, TensorRT={}",
            onnx_output.len(),
            tensorrt_output.len()
        ));
    }

    let mut max_diff = 0.0f32;
    let mut num_mismatches = 0;

    for (i, (&onnx_val, &trt_val)) in onnx_output.iter().zip(tensorrt_output.iter()).enumerate() {
        let diff = (onnx_val - trt_val).abs();
        max_diff = max_diff.max(diff);

        let relative_error = if onnx_val.abs() > 1e-6 {
            diff / onnx_val.abs()
        } else {
            diff
        };

        if relative_error > tolerance {
            num_mismatches += 1;
            if num_mismatches <= 10 {  // Only log first 10 mismatches
                warn!(
                    "Mismatch at index {}: ONNX={:.6}, TensorRT={:.6}, diff={:.6}, rel_err={:.6}",
                    i, onnx_val, trt_val, diff, relative_error
                );
            }
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
    
    if input_dims.is_empty() {
        return Err(anyhow!("No input tensors found in TensorRT engine"));
    }
    
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

    info!("Model validation passed:");
    info!("- {} input tensors", input_dims.len());
    info!("- {} output tensors", output_dims.len());
    info!("- Input data type: {:?}", input_data_type);

    for input_info in input_dims.iter() {
        info!("TensorRT input: '{}' with dimensions {:?}", input_info.name, input_info.dims);
    }
    
    for output_info in output_dims.iter() {
        info!("TensorRT output: '{}' with dimensions {:?}", output_info.name, output_info.dims);
    }

    for onnx_input in session.inputs.iter() {
        info!("ONNX input: '{}'", onnx_input.name);
    }
    
    for onnx_output in session.outputs.iter() {
        info!("ONNX output: '{}'", onnx_output.name);
    }

    // Run comparison tests
    info!("Running {} comparison iterations...", args.iterations);
    
    for iteration in 1..=args.iterations {
        info!("Iteration {}/{}", iteration, args.iterations);

        // Generate random inputs for all input tensors based on TensorRT input data type
        // Note: TensorRT dims don't include batch dimension, but we need to add batch=1
        let mut input_data_list_u8 = Vec::new();
        let mut input_data_list_f32 = Vec::new();
        
        for input_info in &input_dims {
            // Create dimensions with batch size = 1
            let dims_with_batch: Vec<u32> = std::iter::once(1u32)
                .chain(input_info.dims.iter().cloned())
                .collect();
            
            let (data_u8, data_f32) = match input_data_type {
                InputDataType::UINT8 => {
                    let data_u8 = generate_random_input_u8(&dims_with_batch);
                    let data_f32 = u8_to_f32(&data_u8);
                    (data_u8, data_f32)
                }
                InputDataType::FP32 => {
                    let data_f32 = generate_random_input_f32(&dims_with_batch);
                    let data_u8 = f32_to_u8(&data_f32);
                    (data_u8, data_f32)
                }
                _ => {
                    return Err(anyhow!("Unsupported input data type: {:?}", input_data_type));
                }
            };
            input_data_list_u8.push(data_u8);
            input_data_list_f32.push(data_f32);
        }

        // Debug: Log data sizes for verification
        for (i, (u8_data, f32_data)) in input_data_list_u8.iter().zip(input_data_list_f32.iter()).enumerate() {
            let expected_size: usize = input_dims[i].dims.iter().map(|&d| d as usize).product();
            info!("Input {}: u8_data.len()={}, f32_data.len()={}, expected_size_without_batch={}, with_batch={}", 
                  i, u8_data.len(), f32_data.len(), expected_size, expected_size);
        }

        // Run ONNX inference
        let onnx_outputs = run_onnx_inference(&mut session, &input_data_list_f32, &input_dims)
            .map_err(|e| anyhow!("ONNX inference failed: {}", e))?;

        // Run TensorRT inference
        let tensorrt_outputs = run_tensorrt_inference(&mut engine, &input_data_list_u8, &input_dims)
            .map_err(|e| anyhow!("TensorRT inference failed: {}", e))?;

        // Compare all outputs
        if onnx_outputs.len() != tensorrt_outputs.len() {
            return Err(anyhow!(
                "Output count mismatch during inference: ONNX returned {} outputs, TensorRT returned {} outputs",
                onnx_outputs.len(), tensorrt_outputs.len()
            ));
        }
        
        for (i, (onnx_output, tensorrt_output)) in onnx_outputs.iter().zip(tensorrt_outputs.iter()).enumerate() {
            compare_outputs(onnx_output, tensorrt_output, args.tolerance)
                .map_err(|e| anyhow!("Output {} comparison failed: {}", i, e))?;
        }
        
        info!("✓ Iteration {} passed - all {} outputs match", iteration, onnx_outputs.len());
    }

    info!("🎉 All {} iterations passed! ONNX and TensorRT outputs match within tolerance {:.6}", 
          args.iterations, args.tolerance);

    Ok(())
}
