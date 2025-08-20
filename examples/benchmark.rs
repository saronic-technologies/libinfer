//! # Benchmark Example
//!
//! Benchmarks inference speed of TensorRT engines with various batch sizes.
//!
//! ## Usage
//! ```bash
//! # Run with a single engine file
//! cargo run --example benchmark -- --path /path/to/your/model.engine --iterations 100
//!
//! # Run with a directory containing multiple batch sizes
//! cargo run --example benchmark -- --path /directory/with/engines --iterations 100
//! ```
//!
//! ## Engine Requirements
//! This example expects:
//! - When targeting a directory, it looks for: yolov8n.engine, yolov8n_b2.engine, etc.
//! - You must provide your own TensorRT engine files
//! - The example will adapt to any model type, not just YOLOv8
//!

use clap::Parser;
use cxx::UniquePtr;
use libinfer::{Engine, InputDataType, Options};
use libinfer::ffi::TensorInput;
use std::{
    iter::repeat,
    path::PathBuf,
    time::{Duration, Instant},
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

fn benchmark_inference(engine: &mut UniquePtr<Engine>, num_runs: usize) {
    let input_dims = engine.get_input_dims();
    let batch_size = engine.get_batch_dims().opt; // Use optimal batch size from engine
    let input_len = if !input_dims.is_empty() {
        input_dims[0].dims.iter().fold(1, |acc, &e| acc * e as usize) * batch_size as usize
    } else {
        0
    };
    let dtype = engine.get_input_data_type();
    let input_names = engine.get_input_names();
    
    let input_data: Vec<u8> = match dtype {
        InputDataType::UINT8 => repeat(0).take(input_len).collect(),
        InputDataType::FP32 => repeat(0).take(4 * input_len).collect(),
        _ => {
            error!("Unsupported input data type");
            std::process::exit(1);
        },
    };

    // Create input tensors for all inputs
    let input_tensors: Vec<TensorInput> = input_names.iter().map(|name| {
        TensorInput {
            name: name.clone(),
            tensor: input_data.clone(),
        }
    }).collect();

    // Warmup.
    info!("Warming up inference codepath...");
    for _ in 0..1024 {
        let _output = engine.pin_mut().infer(&input_tensors).unwrap();
    }

    // Measure.
    info!("Beginning {num_runs} inference runs...");
    let latencies = (0..num_runs)
        .map(|_| {
            let start = Instant::now();
            let _output = engine.pin_mut().infer(&input_tensors).unwrap();
            start.elapsed()
        })
        .collect::<Vec<Duration>>();

    let total_latency = latencies.iter().map(|t| t.as_secs_f32()).sum::<f32>();
    let average_batch_latency = total_latency / latencies.len() as f32;
    let average_batch_framerate = 1.0 / average_batch_latency;
    let average_frame_latency = total_latency / (latencies.len() as f32 * batch_size as f32);
    let average_frame_framerate = 1.0 / average_frame_latency;

    info!("inference calls    : {}", num_runs);
    info!("total latency      : {}", total_latency);
    info!("avg. frame latency : {}", average_frame_latency);
    info!("avg. frame fps     : {}", average_frame_framerate);
    info!("avg. batch latency : {}", average_batch_latency);
    info!("avg. batch fps     : {}", average_batch_framerate);
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

    // If the exact path was specified, use it directly
    if args.path.is_file() {
        info!("Loading engine from {}", args.path.display());
        let options = Options {
            path: args.path.to_string_lossy().to_string(),
            device_index: 0,
        };
        let mut engine = match Engine::new(&options) {
            Ok(engine) => engine,
            Err(e) => {
                error!("Failed to load engine: {e}");
                std::process::exit(1);
            }
        };

        info!("Input data type: {:?}", engine.get_input_data_type());
        benchmark_inference(&mut engine, args.iterations);
        return;
    }

    // Otherwise, look for engines with different batch sizes
    let b1_path = args.path.join("yolov8n.engine");
    if !b1_path.exists() {
        error!("Engine file not found: {}", b1_path.display());
        error!("Please specify a path to a valid engine file or directory containing engine files");
        std::process::exit(1);
    }

    let b1_options = Options {
        path: b1_path.to_string_lossy().to_string(),
        device_index: 0,
    };
    let mut b1_engine = match Engine::new(&b1_options) {
        Ok(engine) => engine,
        Err(e) => {
            error!("Failed to load engine: {e}");
            std::process::exit(1);
        }
    };

    info!("Input data type: {:?}", b1_engine.get_input_data_type());
    info!("\nRunning benchmark for batch size 1");
    benchmark_inference(&mut b1_engine, args.iterations);

    // Try to load other batch sizes if they exist
    let b2_path = args.path.join("yolov8n_b2.engine");
    if b2_path.exists() {
        let b2_options = Options {
            path: b2_path.to_string_lossy().to_string(),
            device_index: 0,
        };
        match Engine::new(&b2_options) {
            Ok(mut b2_engine) => {
                info!("\nRunning benchmark for batch size 2");
                benchmark_inference(&mut b2_engine, args.iterations / 2);
            },
            Err(e) => error!("Failed to load b2 engine: {e}")
        }
    }

    let b4_path = args.path.join("yolov8n_b4.engine");
    if b4_path.exists() {
        let b4_options = Options {
            path: b4_path.to_string_lossy().to_string(),
            device_index: 0,
        };
        match Engine::new(&b4_options) {
            Ok(mut b4_engine) => {
                info!("\nRunning benchmark for batch size 4");
                benchmark_inference(&mut b4_engine, args.iterations / 4);
            },
            Err(e) => error!("Failed to load b4 engine: {e}")
        }
    }

    let b8_path = args.path.join("yolov8n_b8.engine");
    if b8_path.exists() {
        let b8_options = Options {
            path: b8_path.to_string_lossy().to_string(),
            device_index: 0,
        };
        match Engine::new(&b8_options) {
            Ok(mut b8_engine) => {
                info!("\nRunning benchmark for batch size 8");
                benchmark_inference(&mut b8_engine, args.iterations / 8);
            },
            Err(e) => error!("Failed to load b8 engine: {e}")
        }
    }

    let b16_path = args.path.join("yolov8n_b16.engine");
    if b16_path.exists() {
        let b16_options = Options {
            path: b16_path.to_string_lossy().to_string(),
            device_index: 0,
        };
        match Engine::new(&b16_options) {
            Ok(mut b16_engine) => {
                info!("\nRunning benchmark for batch size 16");
                benchmark_inference(&mut b16_engine, args.iterations / 16);
            },
            Err(e) => error!("Failed to load b16 engine: {e}")
        }
    }
}
