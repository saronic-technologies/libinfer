//! # Benchmark Example
//!
//! Benchmarks inference speed of TensorRT engines with various batch sizes.
//! Supports engines with heterogeneous per-input dynamic shapes.
//!
//! ## Usage
//! ```bash
//! cargo run --release --example benchmark -- --path /path/to/model.engine --iterations 1000
//! ```

use clap::Parser;
use cxx::UniquePtr;
use libinfer::{Engine, TensorDataType, Options};
use libinfer::ffi::InputTensor;
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
use tracing::{info, error, Level};
use tracing_subscriber::{FmtSubscriber, EnvFilter};

#[derive(Parser, Debug)]
struct Args {
    /// Path to the engine file.
    #[arg(short, long, value_name = "PATH", value_parser)]
    path: PathBuf,

    /// Number of iterations (default: 32768)
    #[arg(short, long, value_name = "ITERATIONS", default_value_t = 1 << 15)]
    iterations: usize,

    /// GPU device index
    #[arg(short, long, default_value_t = 0)]
    device: u32,
}

/// Build input tensors using per-input shape profiles at the given phase (min/opt/max).
fn build_inputs(engine: &UniquePtr<Engine>, phase: &str) -> Vec<InputTensor> {
    let profiles = engine.get_input_shape_profiles();
    let input_infos = engine.get_input_dims();

    profiles.iter().zip(input_infos.iter()).map(|(profile, info)| {
        let shape = match phase {
            "min" => &profile.min_shape,
            "opt" => &profile.opt_shape,
            "max" => &profile.max_shape,
            _ => &profile.opt_shape,
        };

        let dtype_size: usize = match info.dtype {
            TensorDataType::UINT8 | TensorDataType::BOOL => 1,
            TensorDataType::FP32 => 4,
            TensorDataType::INT64 => 8,
            _ => 1,
        };

        let elem_count: usize = shape.iter().map(|&d| d as usize).product();
        let dynamic_tag = if profile.has_dynamic_shape { "DYNAMIC" } else { "STATIC" };

        info!("  {} '{}': shape={:?} ({} bytes)",
             dynamic_tag, profile.name, shape, elem_count * dtype_size);

        InputTensor {
            name: info.name.clone(),
            data: vec![0u8; elem_count * dtype_size],
            dtype: info.dtype.clone(),
        }
    }).collect()
}

fn benchmark_inference(engine: &mut UniquePtr<Engine>, input_tensors: &Vec<InputTensor>, num_runs: usize) {
    // Warmup
    info!("Warming up (1024 iterations)...");
    for _ in 0..1024 {
        let _ = engine.pin_mut().infer(input_tensors).unwrap();
    }

    // Measure
    info!("Running {num_runs} iterations...");
    let mut latencies = Vec::with_capacity(num_runs);
    let mut total_time = Duration::ZERO;

    for i in 0..num_runs {
        let start = Instant::now();
        let _output = engine.pin_mut().infer(input_tensors).unwrap();
        let elapsed = start.elapsed();
        latencies.push(elapsed);
        total_time += elapsed;

        let progress_interval = std::cmp::min(num_runs / 10, 1000).max(1);
        if i % progress_interval == 0 && i > 0 {
            let avg_ms = total_time.as_secs_f64() / i as f64 * 1000.0;
            let remaining = num_runs - i;
            let eta = avg_ms * remaining as f64 / 1000.0;
            info!("  {}/{} ({:.1}%) avg={:.3}ms ETA={:.1}s",
                  i, num_runs, (i as f64 / num_runs as f64) * 100.0, avg_ms, eta);
        }
    }

    latencies.sort();
    let total_secs = total_time.as_secs_f64();
    let avg_ms = total_secs / num_runs as f64 * 1000.0;
    let p50_ms = latencies[num_runs / 2].as_secs_f64() * 1000.0;
    let p99_ms = latencies[num_runs * 99 / 100].as_secs_f64() * 1000.0;
    let min_ms = latencies[0].as_secs_f64() * 1000.0;
    let max_ms = latencies[num_runs - 1].as_secs_f64() * 1000.0;

    info!("Results:");
    info!("  iterations : {}", num_runs);
    info!("  avg        : {:.3}ms", avg_ms);
    info!("  p50        : {:.3}ms", p50_ms);
    info!("  p99        : {:.3}ms", p99_ms);
    info!("  min        : {:.3}ms", min_ms);
    info!("  max        : {:.3}ms", max_ms);
    info!("  throughput : {:.1} infer/s", num_runs as f64 / total_secs);
}

fn main() {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");

    let args = Args::parse();

    if !args.path.is_file() {
        error!("Engine file not found: {}", args.path.display());
        std::process::exit(1);
    }

    info!("Loading engine: {}", args.path.display());
    let options = Options {
        path: args.path.to_string_lossy().to_string(),
        device_index: args.device,
    };

    let mut engine = Engine::new(&options).unwrap_or_else(|e| {
        error!("Failed to load engine: {e}");
        std::process::exit(1);
    });

    // Display shape profiles
    let profiles = engine.get_input_shape_profiles();
    info!("Inputs: {}", profiles.len());
    for p in &profiles {
        if p.has_dynamic_shape {
            info!("  '{}': DYNAMIC min={:?} opt={:?} max={:?}",
                 p.name, p.min_shape, p.opt_shape, p.max_shape);
        } else {
            info!("  '{}': STATIC shape={:?}", p.name, p.min_shape);
        }
    }

    let output_infos = engine.get_output_dims();
    info!("Outputs: {}", output_infos.len());
    for o in &output_infos {
        info!("  '{}': dims={:?} dtype={:?}", o.name, o.dims, o.dtype);
    }

    // Check if any inputs are dynamic
    let has_dynamic = profiles.iter().any(|p| p.has_dynamic_shape);

    if has_dynamic {
        // Benchmark at min, opt, and max shapes
        for phase in &["min", "opt", "max"] {
            info!("\n=== Benchmark at {} shapes ===", phase);
            let inputs = build_inputs(&engine, phase);
            benchmark_inference(&mut engine, &inputs, args.iterations);
        }
    } else {
        // Static engine — single benchmark
        info!("\n=== Benchmark (static shapes) ===");
        let inputs = build_inputs(&engine, "opt");
        benchmark_inference(&mut engine, &inputs, args.iterations);
    }
}
