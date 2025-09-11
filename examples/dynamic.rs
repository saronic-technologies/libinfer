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
    // Pinned host memory improves transfer overlap
    let _ = engine.pin_mut().enable_pinned_memory(true);

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

        // Helper to construct one set of inputs for this batch size
        let build_inputs = |infos: &Vec<libinfer::TensorInfo>| -> Vec<TensorInstance> {
            infos.iter().map(|info| {
                let new_shape: Vec<i64> = info.shape.iter().map(|&d| if d == -1 { batch_size } else { d }).collect();
                let elems = new_shape.iter().fold(1usize, |acc, &e| acc * (e as usize));
                match info.dtype {
                    TensorDataType::UINT8 | TensorDataType::BOOL =>
                        TensorInstance::from_u8(info.name.clone(), new_shape, vec![0u8; elems]),
                    TensorDataType::FP32 =>
                        TensorInstance::from_f32(info.name.clone(), new_shape, vec![0f32; elems]),
                    TensorDataType::INT64 =>
                        TensorInstance::from_i64(info.name.clone(), new_shape, vec![0i64; elems]),
                    _ => {
                        error!("Unsupported data type: {:?}", info.dtype);
                        std::process::exit(1);
                    }
                }
            }).collect()
        };

        // Double-buffer to avoid re-register warnings when using pinned memory
        let input_tensors_a = build_inputs(&input_infos);
        let input_tensors_b = build_inputs(&input_infos);

        info!("Total input size across all tensors: {} bytes", input_tensors_a.iter().map(|t| t.data.len()).sum::<usize>());

        // Warmup (alternate buffers) and enable CUDA graphs for hot path
        for i in 0..32 {
            let _ = if i % 2 == 0 { engine.pin_mut().infer(&input_tensors_a) } else { engine.pin_mut().infer(&input_tensors_b) };
        }
        let _ = engine.pin_mut().enable_cuda_graphs();
        let _ = engine.pin_mut().set_validation_enabled(false);

        // Measure inference time
        let start = Instant::now();
        let result = engine.pin_mut().infer(&input_tensors_a);
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
        // Re-enable validation for next shape (optional safety)
        let _ = engine.pin_mut().set_validation_enabled(true);
    }
}
