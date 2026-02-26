
//! # sync_compare
//!
//! Directly compares inference latency with and without the pre-enqueue
//! cudaStreamSynchronize, using the same engine and the same input data.
//!
//! ## Usage
//! ```bash
//! cargo run --example sync_compare -- --engine path/to/model.engine
//! cargo run --example sync_compare -- --engine path/to/model.engine --warmup 500 --iterations 2000
//! ```

use clap::Parser;
use libinfer::{Engine, InputTensor, Options, TensorDataType};
use std::time::{Duration, Instant};
use tracing::info;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser)]
struct Args {
    /// Path to TensorRT engine file
    #[arg(short, long)]
    engine: String,

    /// GPU device index
    #[arg(short, long, default_value_t = 0)]
    device: u32,

    /// Warmup iterations (run before timing to stabilize GPU clocks)
    #[arg(short, long, default_value_t = 200)]
    warmup: usize,

    /// Timed iterations per variant
    #[arg(short, long, default_value_t = 1000)]
    iterations: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted[idx]
}

fn run_timed(
    engine: &mut cxx::UniquePtr<Engine>,
    inputs: &Vec<InputTensor>,
    iterations: usize,
    with_sync: bool,
) -> Vec<Duration> {
    let mut latencies = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        if with_sync {
            engine.pin_mut().infer_with_sync(inputs).unwrap();
        } else {
            engine.pin_mut().infer(inputs).unwrap();
        }
        latencies.push(start.elapsed());
    }
    latencies
}

fn print_stats(label: &str, latencies: &[Duration]) {
    let mut micros: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1e6).collect();
    micros.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = micros.iter().sum::<f64>() / micros.len() as f64;
    let p50 = percentile(&micros, 50.0);
    let p95 = percentile(&micros, 95.0);
    let p99 = percentile(&micros, 99.0);
    let min = micros[0];
    let max = *micros.last().unwrap();

    println!(
        "{:<14}  mean={:>7.1}µs  p50={:>7.1}µs  p95={:>7.1}µs  p99={:>7.1}µs  min={:>7.1}µs  max={:>7.1}µs",
        label, mean, p50, p95, p99, min, max
    );
}

fn main() {
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )
    .unwrap();

    let args = Args::parse();

    let mut engine = Engine::new(&Options {
        path: args.engine.clone(),
        device_index: args.device,
    })
    .expect("Failed to load engine");

    let input_dims = engine.get_input_dims();
    let batch_size = engine.get_batch_dims().opt as usize;

    let inputs: Vec<InputTensor> = input_dims
        .iter()
        .map(|t| {
            let elems: usize = t.dims.iter().map(|&d| d as usize).product::<usize>().max(1);
            let bytes_per_elem = match t.dtype {
                TensorDataType::FP32 => 4,
                TensorDataType::INT64 => 8,
                _ => 1,
            };
            InputTensor {
                name: t.name.clone(),
                data: vec![0u8; elems * batch_size * bytes_per_elem],
                dtype: t.dtype.clone(),
            }
        })
        .collect();

    info!(
        "Engine: {}  batch={}  inputs={}",
        args.engine,
        batch_size,
        inputs.len()
    );

    // Warmup both paths equally so GPU state is identical going into each timed run.
    info!("Warming up ({} iters each)...", args.warmup);
    run_timed(&mut engine, &inputs, args.warmup / 2, false);
    run_timed(&mut engine, &inputs, args.warmup / 2, true);

    info!("Timing without sync ({} iters)...", args.iterations);
    let no_sync = run_timed(&mut engine, &inputs, args.iterations, false);

    info!("Timing with sync ({} iters)...", args.iterations);
    let with_sync = run_timed(&mut engine, &inputs, args.iterations, true);

    println!();
    println!("=== Results ({} iterations) ===", args.iterations);
    print_stats("without sync", &no_sync);
    print_stats("with sync   ", &with_sync);

    let mean_no_sync: f64 = no_sync.iter().map(|d| d.as_secs_f64() * 1e6).sum::<f64>()
        / no_sync.len() as f64;
    let mean_with_sync: f64 = with_sync.iter().map(|d| d.as_secs_f64() * 1e6).sum::<f64>()
        / with_sync.len() as f64;
    let overhead = mean_with_sync - mean_no_sync;

    println!();
    println!("sync overhead (mean): {:.1}µs per call", overhead);
}
