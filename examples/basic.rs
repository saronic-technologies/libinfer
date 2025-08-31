//! # Basic Example
//!
//! Demonstrates basic functionality of `libinfer` by running inference
//! on a TensorRT engine with a synthetic input.
//!
//! ## Usage
//! ```bash
//! cargo run --example basic -- --path /path/to/your/model.engine
//! ```
//!
//! ## Engine Requirements
//! - You must provide your own TensorRT engine file (.engine)
//! - This example works with any TensorRT engine
//! - The example creates zero-filled synthetic input data with the correct dimensions
//! - To create engine files, use the TensorRT Python API or trtexec command-line tool

use clap::Parser;
use libinfer::{Engine, TensorDataType, Options, TensorInstance};
use std::path::PathBuf;
use tracing::{info, error, Level};
use tracing_subscriber::{FmtSubscriber, EnvFilter};

#[derive(Parser, Debug)]
#[clap(about = "Basic example for libinfer")]
struct Args {
    /// Path to the engine file
    #[arg(short, long, value_name = "PATH", value_parser)]
    path: PathBuf,

    /// Number of iterations to run
    #[arg(short, long, value_name = "ITERATIONS", default_value_t = 1 << 10)]
    iterations: usize,

    /// GPU device index to use
    #[arg(short, long, value_name = "DEVICE", default_value_t = 0)]
    device: u32,
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

    info!("Loading TensorRT engine from: {}", args.path.display());

    // Create engine options
    let options = Options {
        path: args.path.to_string_lossy().to_string(),
        device_index: args.device,
    };

    // Load the engine
    let mut engine = Engine::new(&options).unwrap_or_else(|e| {
        error!("Failed to load engine: {e}");
        std::process::exit(1);
    });

    let input_infos = engine.get_input_tensor_info();
    let output_infos = engine.get_output_tensor_info();

    // Print model information
    info!("Engine loaded successfully");
    info!("Number of inputs: {}", input_infos.len());
    info!("Number of outputs: {}", output_infos.len());
    info!("Engine supports dynamic shapes");
    
    // Print detailed information for all input tensors
    info!("Input tensors:");
    for input_info in &input_infos {
        info!("  '{}': {:?} {:?}", input_info.name, input_info.shape, input_info.dtype);
    }
    
    // Print detailed information for all output tensors
    info!("Output tensors:");
    for output_info in &output_infos {
        info!("  '{}': {:?} {:?}", output_info.name, output_info.shape, output_info.dtype);
    }

    // Create input tensors for all inputs
    let mut input_tensors = Vec::new();
    
    for input_info in &input_infos {
        // Calculate tensor size from shape (add batch dimension of 1)
        let shape_with_batch: Vec<i64> = std::iter::once(1i64)
            .chain(input_info.shape.iter().cloned())
            .collect();
        let input_size = shape_with_batch.iter().fold(1, |acc, &e| acc * e as usize);

        // Create appropriate input data based on data type
        let input_data = match input_info.dtype {
            TensorDataType::UINT8 => vec![0u8; input_size],
            TensorDataType::FP32 => {
                // For FP32, we need 4 bytes per element
                vec![0u8; input_size * 4]
            }
            TensorDataType::INT64 => {
                // For INT64, we need 8 bytes per element
                vec![0u8; input_size * 8]
            }
            TensorDataType::BOOL => vec![0u8; input_size],
            _ => {
                error!("Unsupported input data type");
                std::process::exit(1);
            }
        };

        input_tensors.push(TensorInstance {
            name: input_info.name.clone(),
            data: input_data,
            shape: shape_with_batch,
            dtype: input_info.dtype.clone(),
        });
    }

    info!("Running inference for {} iterations...", args.iterations);

    // Run inference for specified number of iterations
    for i in 0..args.iterations {
        if i % (args.iterations / 10).max(1) == 0 {
            info!("Iteration {}/{}", i, args.iterations);
        }

        let result = engine.pin_mut().infer(&input_tensors);

        match result {
            Ok(outputs) => {
                if i == 0 {
                    // Print output information on first iteration
                    info!("Inference successful! Output tensors:");
                    for output in &outputs {
                        info!("  '{}' type {:?} : {} elements", output.name, output.dtype, output.data.len());
                    }
                }
            }
            Err(e) => {
                error!("Inference error: {e}");
                break;
            }
        }
    }

    info!("Inference complete!");
}
