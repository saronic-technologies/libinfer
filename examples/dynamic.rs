//! # Dynamic Batch Size Example
//!
//! Demonstrates using a TensorRT engine with dynamic batch sizes.
//! This example shows how to:
//! 1. Check the supported batch size range of an engine
//! 2. Create inputs with different batch sizes
//! 3. Run inference with varying batch sizes
//!
//! ## Usage
//! ```bash
//! cargo run --example dynamic -- --path /path/to/your/dynamic_batch.engine
//! ```
//!
//! ## Engine Requirements
//! - You must provide your own TensorRT engine file
//! - The engine must be built with dynamic batch size support
//! - Normal fixed-batch engines will display a warning message
//! - To create a dynamic batch engine, use TensorRT's Python API or trtexec with appropriate flags:
//!   ```
//!   trtexec --onnx=model.onnx --saveEngine=dynamic_model.engine --minShapes=input:1x3x640x640 --optShapes=input:8x3x640x640 --maxShapes=input:16x3x640x640
//!   ```

use clap::Parser;
use libinfer::{Engine, TensorDataType, Options, TensorInstance};
use std::{path::PathBuf, time::Instant};
use tracing::{info, warn, error, Level};
use tracing_subscriber::{FmtSubscriber, EnvFilter};

#[derive(Parser, Debug)]
#[clap(about = "Dynamic batch size example for libinfer")]
struct Args {
    /// Path to the engine file with dynamic batch support
    #[arg(short, long, value_name = "PATH", value_parser)]
    path: PathBuf,

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

    // Check if this engine has dynamic dimensions
    let has_dynamic = input_infos.iter().any(|info| info.shape.contains(&-1));
    if !has_dynamic {
        warn!("This engine does not appear to have dynamic dimensions!");
        warn!("To test dynamic shapes, you need an engine built with dynamic dimensions.");
        warn!("Continuing with fixed shapes...");
    }

    // Test different batch sizes (using typical values)
    let batch_sizes_to_test = [1, 4, 8];

    for &batch_size in &batch_sizes_to_test {
        info!("\nTesting batch size: {}", batch_size);

        // Create input tensors for all inputs
        let input_tensors: Vec<TensorInstance> = input_infos.iter().map(|info| {
            let dtype_size = match info.dtype {
                TensorDataType::UINT8 => 1,
                TensorDataType::FP32 => 4,
                TensorDataType::INT64 => 8,
                TensorDataType::BOOL => 1,
                _ => {
                    error!("Unsupported data type: {:?}", info.dtype);
                    1 // Default to 1 byte to avoid panic
                }
            };

            // Calculate tensor size from shape use batch_size for all dynamic dimensions (which are -1) 
            let new_shape: Vec<i64> = info.shape.iter().map(|&d| if d == -1 { batch_size } else { d }).collect();
            let input_size = new_shape.iter().fold(1, |acc, &e| {
                let e = if e == -1 { 1 } else { e };
                acc * e as usize
            });

            TensorInstance {
                name: info.name.clone(),
                data: vec![0u8; input_size * dtype_size],
                shape: new_shape,
                dtype: info.dtype.clone(),
            }
        }).collect();

        info!("Total input size across all tensors: {} bytes", input_tensors.iter().map(|t| t.data.len()).sum::<usize>());

        // Warmup
        for _ in 0..5 {
            let _ = engine.pin_mut().infer(&input_tensors);
        }

        // Measure inference time
        let start = Instant::now();
        let result = engine.pin_mut().infer(&input_tensors);
        let elapsed = start.elapsed();

        match result {
            Ok(outputs) => {
                info!("Inference successful!");
                info!("Number of output tensors: {}", outputs.len());
                for (i, output) in outputs.iter().enumerate() {
                    info!("Output {}: '{}' with {} elements", i, output.name, output.data.len());
                }
                info!("Inference time: {:?}", elapsed);
                info!("Throughput: {:.2} items/second",
                     batch_size as f64 / elapsed.as_secs_f64());
            }
            Err(e) => {
                error!("Inference failed: {}", e);
            }
        }
    }
}
