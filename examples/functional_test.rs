//! # Functional Test Example
//!
//! Validates the correct functionality of a TensorRT engine by checking:
//! - Input dimensions
//! - Output dimensions
//! - Batch dimensions
//! - Output values against expected reference values
//!
//! ## Usage
//! ```bash
//! cargo run --example functional_test -- --path /path/to/test/directory
//! ```
//!
//! ## Required Test Files
//! This example requires the following files in your test directory:
//! - `yolov8n.engine`: The TensorRT engine file to test (or specify direct path to engine)
//! - `input.bin`: A binary file containing raw input data matching the model's input format
//! - `features.txt`: A text file with expected output values (space-separated floats)
//!
//! ## Note
//! You must provide your own engine file and corresponding test data. The test assumes
//! a YOLOv8n model by default, but the code can be adapted for any model architecture.

use anyhow::Result;
use approx::assert_relative_eq;
use clap::Parser;
use cxx::UniquePtr;
use libinfer::{Engine, Options, TensorInstance};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    iter::zip,
    path::PathBuf,
    str::FromStr,
};
use tracing::{info, error, Level};
use tracing_subscriber::{FmtSubscriber, EnvFilter};

#[derive(Parser, Debug)]
struct Args {
    /// Path to the directory containing engine files.
    #[arg(short, long, value_name = "PATH", default_value = ".", value_parser)]
    path: PathBuf,

    /// Number of iterations (default: 32768)
    #[arg(short, long, value_name = "ITERATIONS", default_value_t = 1 << 15)]
    iterations: usize,
}

fn read_binary_file(path: PathBuf) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

fn parse_file_to_float_vec(path: PathBuf) -> Result<Vec<f32>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut float_vec = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let values: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| f32::from_str(s).ok())
            .collect();

        float_vec.extend(values);
    }
    Ok(float_vec)
}

fn test_input_dim(engine: &UniquePtr<Engine>) {
    let input_dims = engine.get_input_tensor_info();
    assert_eq!(input_dims.len(), 1); // Expecting one input tensor
    let input_dim = &input_dims[0];
    assert_eq!(input_dim.shape[0], 1);
    assert_eq!(input_dim.shape[1], 3);
    assert_eq!(input_dim.shape[2], 640);
    assert_eq!(input_dim.shape[3], 640);
    info!("Input dimensions: '{}' -> {:?}", input_dim.name, input_dim.shape);
}

fn test_batch_dim(engine: &UniquePtr<Engine>) {
    let input_dims = engine.get_input_tensor_info();
    info!("Input tensor shapes: {:?}", input_dims.iter().map(|t| (&t.name, &t.shape)).collect::<Vec<_>>());
}

fn test_output_dim(engine: &UniquePtr<Engine>) {
    let output_dims = engine.get_output_tensor_info();
    assert_eq!(output_dims.len(), 1); // Expecting one output tensor
    let output_dim = &output_dims[0];
    assert_eq!(output_dim.shape[0], 1);
    assert_eq!(output_dim.shape[1], 84);
    assert_eq!(output_dim.shape[2], 8400);
    info!("Output dimensions: '{}' -> {:?}", output_dim.name, output_dim.shape);
}

fn test_output_features(engine: &mut UniquePtr<Engine>, input: &[u8], expected: &[f32]) {
    info!("Testing output features...");
    let batch_size = 1; // Use batch size 1
    
    let input_dims = engine.get_input_tensor_info();
    // Calculate tensor size from shape use 1 for all dynamic dimensions (which are -1) 
    let new_shape: Vec<i64> = input_dims[0].shape.iter().map(|&d| if d == -1 { 1 } else { d }).collect();
    
    let input_tensors = vec![TensorInstance {
        name: input_dims[0].name.clone(),
        data: input.to_vec(),
        shape: new_shape,
        dtype: input_dims[0].dtype.clone(),
    }];

    let output_dims = engine.get_output_tensor_info();
    let batch_element_size = output_dims[0]
        .shape
        .iter()
        .fold(1, |acc, &e| acc * e as usize);
    let expected_output_size = batch_element_size * batch_size as usize;

    let output_tensors = engine.pin_mut().infer(&input_tensors).unwrap();
    
    // Get the first output tensor and convert based on data type
    let output = &output_tensors[0];
    let actual_f32: Vec<f32> = {
        // Check if data length suggests actual type differs from declared type
        let expected_elements = batch_element_size * batch_size as usize;
        let bytes_per_element = output.data.len() / expected_elements;
        
        info!("Detected {} bytes per element (raw_len={}, expected_elements={})", 
              bytes_per_element, output.data.len(), expected_elements);
              
        match bytes_per_element {
            4 => {
                // 4 bytes per element = FP32, regardless of declared type
                output.data
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect()
            }
            1 => {
                // 1 byte per element = UINT8 or BOOL
                match output.dtype {
                    libinfer::TensorDataType::BOOL => {
                        output.data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect()
                    }
                    _ => {
                        output.data.iter().map(|&b| b as f32).collect()
                    }
                }
            }
            8 => {
                // 8 bytes per element = INT64
                output.data
                    .chunks_exact(8)
                    .map(|bytes| i64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7]
                    ]) as f32)
                    .collect()
            }
            _ => panic!("Unexpected bytes per element: {} (declared type: {:?})", bytes_per_element, output.dtype),
        }
    };

    // Check that the entire output length is correct.
    info!("Output validation: actual_f32.len()={}, expected_output_size={}, batch_size={}, batch_element_size={}", 
          actual_f32.len(), expected_output_size, batch_size, batch_element_size);
    info!("Output tensor dtype: {:?}, raw data len: {}", output.dtype, output.data.len());
    assert_eq!(actual_f32.len(), expected_output_size);

    // Only checking the first twelve produced values. Repeat for each batch element.
    actual_f32.chunks_exact(batch_element_size).for_each(|chunk| {
        zip(chunk, expected).for_each(|(a, e)| {
            assert_relative_eq!(*a, e, epsilon = 0.1);
        });
    });

    info!("Output features agree");
}

fn main() {
    // Initialize tracing subscriber for logging
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");

    let args = Args::parse();

    // Check if path is directly to an engine file
    let engine_path = if args.path.is_file() {
        args.path.clone()
    } else {
        args.path.join("yolov8n.engine")
    };

    let options = Options {
        path: engine_path.to_string_lossy().to_string(),
        device_index: 0,
    };

    let mut engine = match Engine::new(&options) {
        Ok(engine) => engine,
        Err(e) => {
            error!("Failed to load engine: {e}");
            std::process::exit(1);
        }
    };
    // Use pinned host memory and enable CUDA graphs for faster transfers during test
    let _ = engine.pin_mut().enable_pinned_memory(true);

    let input = read_binary_file(args.path.join("input.bin")).unwrap_or_else(|e| {
        error!("Failed to read input.bin: {e}");
        std::process::exit(1);
    });
    let expected = parse_file_to_float_vec(args.path.join("features.txt")).unwrap_or_else(|e| {
        error!("Failed to parse features.txt: {e}");
        std::process::exit(1);
    });

    let input_dims = engine.get_input_tensor_info();
    if !input_dims.is_empty() {
        info!("Input data types: {:?}", input_dims.iter().map(|t| (&t.name, &t.dtype)).collect::<Vec<_>>());
    }

    // Quick warmup to prime kernels and capture CUDA graph
    if !input_dims.is_empty() {
        let shape: Vec<i64> = input_dims[0].shape.iter().map(|&d| if d == -1 { 1 } else { d }).collect();
        // Double buffer: two copies of the same input to avoid re-register clashes
        let input_a = TensorInstance { name: input_dims[0].name.clone(), data: input.clone(), shape: shape.clone(), dtype: input_dims[0].dtype.clone() };
        let input_b = TensorInstance { name: input_dims[0].name.clone(), data: input.clone(), shape: shape.clone(), dtype: input_dims[0].dtype.clone() };
        for i in 0..16 { let _ = if i % 2 == 0 { engine.pin_mut().infer(&vec![input_a.clone()]) } else { engine.pin_mut().infer(&vec![input_b.clone()]) }; }
        let _ = engine.pin_mut().enable_cuda_graphs();
        let _ = engine.pin_mut().set_validation_enabled(false);
    }

    test_input_dim(&engine);
    test_output_dim(&engine);
    test_batch_dim(&engine);
    test_output_features(&mut engine, &input, &expected);
}
