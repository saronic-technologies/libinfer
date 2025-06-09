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
use libinfer::{Engine, InputDataType, Options};
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

    // Print model information
    info!("Engine loaded successfully");
    info!("Input dimensions: {:?}", engine.get_input_dims());
    info!("Output dimensions: {:?}", engine.get_output_dims());
    info!("Batch dimensions: {:?}", engine.get_batch_dims());
    info!("Input data type: {:?}", engine.get_input_data_type());

    // Create input data based on input dimensions and data type
    let input_dims = engine.get_input_dims();
    let input_size = input_dims.iter().fold(1, |acc, &e| acc * e as usize);

    // Create appropriate input based on data type
    let input = match engine.get_input_data_type() {
        InputDataType::UINT8 => vec![0u8; input_size],
        InputDataType::FP32 => {
            // For FP32, we need 4 bytes per element
            vec![0u8; input_size * 4]
        }
        _ => {
            error!("Unsupported input data type");
            std::process::exit(1);
        }
    };

    info!("Running inference for {} iterations...", args.iterations);

    // Run inference for specified number of iterations
    for i in 0..args.iterations {
        if i % (args.iterations / 10).max(1) == 0 {
            info!("Iteration {}/{}", i, args.iterations);
        }

        let result = engine.pin_mut().infer(&input);

        if let Err(e) = &result {
            error!("Inference error: {e}");
            break;
        }
    }

    info!("Inference complete!");
}
