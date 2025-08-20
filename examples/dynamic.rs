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
use libinfer::{Engine, InputDataType, Options};
use libinfer::ffi::InputTensor;
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

    // Print model information
    let batch_dims = engine.get_batch_dims();
    info!("Engine loaded successfully");
    info!("Number of inputs: {}", engine.get_num_inputs());
    info!("Number of outputs: {}", engine.get_num_outputs());
    info!("Input names: {:?}", engine.get_input_names());
    info!("Output names: {:?}", engine.get_output_names());
    info!("Input dimensions: {:?}", engine.get_input_dims());
    info!("Output dimensions: {:?}", engine.get_output_dims());
    info!("Batch dimensions: min={}, optimal={}, max={}",
         batch_dims.min, batch_dims.opt, batch_dims.max);
    info!("Input data type: {:?}", engine.get_input_data_type());

    // Check if this engine truly supports dynamic batch sizes
    if batch_dims.min == batch_dims.max {
        warn!("This engine does not support dynamic batch sizes!");
        warn!("All batch dimensions are fixed at {}", batch_dims.min);
        warn!("To test dynamic batching, you need an engine built with dynamic shapes.");
        return;
    }

    // Create input data based on input dimensions and data type
    let input_dims = engine.get_input_dims();
    let input_size_per_item = if !input_dims.is_empty() {
        input_dims[0].dims.iter().fold(1, |acc, &e| acc * e as usize)
    } else {
        0
    };
    let input_names = engine.get_input_names();

    // Test different batch sizes within the supported range
    let batch_sizes_to_test = [
        batch_dims.min,
        batch_dims.opt,
        batch_dims.max,
    ];

    for &batch_size in &batch_sizes_to_test {
        info!("\nTesting batch size: {}", batch_size);

        // Create input with the current batch size
        let total_elements = input_size_per_item * batch_size as usize;

        // Create appropriate input based on data type
        let input_data = match engine.get_input_data_type() {
            InputDataType::UINT8 => vec![0u8; total_elements],
            InputDataType::FP32 => {
                // For FP32, we need 4 bytes per element
                vec![0u8; total_elements * 4]
            }
            _ => {
                error!("Unsupported input data type");
                std::process::exit(1);
            }
        };

        // Create input tensors for all inputs
        let input_tensors: Vec<InputTensor> = input_names.iter().map(|name| {
            InputTensor {
                name: name.clone(),
                data: input_data.clone(),
            }
        }).collect();

        info!("Input size: {} elements", total_elements);

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
