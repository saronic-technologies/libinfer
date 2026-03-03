//! # Dynamic Batch Size Example
//!
//! Demonstrates using a TensorRT engine with per-input dynamic batch sizes.
//! This example shows how to:
//! 1. Query per-input shape profiles to see which inputs are dynamic
//! 2. Create inputs with independent batch sizes per tensor
//! 3. Run inference with heterogeneous dynamic batches
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
//! - To create a heterogeneous dynamic batch engine, use trtexec with per-input shapes:
//!   ```
//!   trtexec --onnx=model.onnx --saveEngine=model.engine \
//!     --minShapes=frame:1x3x518x518,crops:1x3x224x224,modality:1 \
//!     --optShapes=frame:1x3x518x518,crops:16x3x224x224,modality:1 \
//!     --maxShapes=frame:1x3x518x518,crops:64x3x224x224,modality:1
//!   ```

use clap::Parser;
use libinfer::{Engine, TensorDataType, Options};
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
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");

    let args = Args::parse();

    info!("Loading TensorRT engine from: {}", args.path.display());

    let options = Options {
        path: args.path.to_string_lossy().to_string(),
        device_index: args.device,
    };

    let mut engine = Engine::new(&options).unwrap_or_else(|e| {
        error!("Failed to load engine: {e}");
        std::process::exit(1);
    });

    let input_infos = engine.get_input_dims();
    let output_infos = engine.get_output_dims();
    let profiles = engine.get_input_shape_profiles();

    info!("Engine loaded successfully");
    info!("Number of inputs: {}", input_infos.len());
    info!("Number of outputs: {}", output_infos.len());

    // Display per-input shape profiles
    let mut has_dynamic = false;
    for profile in &profiles {
        if profile.has_dynamic_shape {
            has_dynamic = true;
            info!("Input '{}': DYNAMIC min={:?} opt={:?} max={:?}",
                 profile.name, profile.min_shape, profile.opt_shape, profile.max_shape);
        } else {
            info!("Input '{}': STATIC shape={:?}",
                 profile.name, profile.min_shape);
        }
    }

    if !has_dynamic {
        warn!("This engine has no dynamic inputs. Nothing to test.");
        return;
    }

    // Test with min, opt, and max dynamic dim values
    for phase in &["min", "opt", "max"] {
        info!("\n--- Testing with {} shapes ---", phase);

        // Build per-input tensors using each input's own profile
        let input_tensors: Vec<InputTensor> = profiles.iter().zip(input_infos.iter()).map(|(profile, info)| {
            let shape = match *phase {
                "min" => &profile.min_shape,
                "opt" => &profile.opt_shape,
                "max" => &profile.max_shape,
                _ => unreachable!(),
            };

            let dtype_size = match info.dtype {
                TensorDataType::UINT8 => 1,
                TensorDataType::FP32 => 4,
                TensorDataType::INT64 => 8,
                TensorDataType::BOOL => 1,
                _ => 1,
            };

            let elem_count: usize = shape.iter().map(|&d| d as usize).product();

            info!("  Input '{}': shape={:?} ({} bytes)",
                 profile.name, shape, elem_count * dtype_size);

            InputTensor {
                name: info.name.clone(),
                data: vec![0u8; elem_count * dtype_size],
                dtype: info.dtype.clone(),
            }
        }).collect();

        // Warmup
        for _ in 0..5 {
            let _ = engine.pin_mut().infer(&input_tensors);
        }

        let start = Instant::now();
        let result = engine.pin_mut().infer(&input_tensors);
        let elapsed = start.elapsed();

        match result {
            Ok(outputs) => {
                info!("Inference successful! ({:?})", elapsed);
                for output in outputs.iter() {
                    info!("  Output '{}': {} bytes (dtype: {:?})",
                         output.name, output.data.len(), output.dtype);
                }
            }
            Err(e) => {
                error!("Inference failed: {}", e);
            }
        }
    }
}
