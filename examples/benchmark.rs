//! Benchmarks inference speed of a TensorRT engine.
//!
//! ## Usage
//! ```bash
//! cargo run --example benchmark -- --path /path/to/your/model.engine --iterations 100
//! ```

use clap::Parser;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use libinfer::{Engine, Options};
use std::path::PathBuf;
use std::process;
use std::time::{Duration, Instant};
use tracing::{error, info, Level};
use tracing::subscriber::set_global_default;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, value_name = "PATH", value_parser)]
    path: PathBuf,

    #[arg(short, long, value_name = "ITERATIONS", default_value_t = 1 << 15)]
    iterations: usize,

    #[arg(short, long, value_name = "DEVICE", default_value_t = 0)]
    device: u32,
}

fn main() {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(Level::INFO)
        .finish();
    set_global_default(subscriber).expect("Failed to set tracing subscriber");

    let args = Args::parse();

    if !args.path.is_file() {
        error!("Engine file not found: {}", args.path.display());
        process::exit(1);
    }

    info!("Loading engine from {}", args.path.display());
    let ctx = CudaContext::new(args.device as usize).expect("failed to create CUDA context");
    unsafe { ctx.disable_event_tracking() };

    let options = Options {
        path: args.path.to_string_lossy().to_string(),
    };

    let mut engine = Engine::new(&options).unwrap_or_else(|e| {
        error!("Failed to load engine: {e}");
        process::exit(1);
    });

    let input_infos = engine.get_input_dims();
    let output_infos = engine.get_output_dims();
    let batch_size = engine.get_batch_dims().opt;

    let stream = ctx.new_stream().expect("failed to create stream");

    let input_bufs: Vec<_> = input_infos
        .iter()
        .map(|ti| {
            stream
                .alloc_zeros::<u8>(ti.byte_size() * batch_size as usize)
                .expect("alloc failed")
        })
        .collect();

    let mut output_bufs: Vec<_> = output_infos
        .iter()
        .map(|ti| {
            stream
                .alloc_zeros::<u8>(ti.byte_size() * batch_size as usize)
                .expect("alloc failed")
        })
        .collect();

    let input_refs: Vec<_> = input_bufs.iter().map(|b| b.device_ptr(&stream)).collect();
    let output_refs: Vec<_> = output_bufs
        .iter_mut()
        .map(|b| b.device_ptr_mut(&stream))
        .collect();
    let input_ptrs: Vec<_> = input_refs.iter().map(|(ptr, _)| *ptr).collect();
    let output_ptrs: Vec<_> = output_refs.iter().map(|(ptr, _)| *ptr).collect();

    info!("Warming up...");
    for _ in 0..1024 {
        engine
            .infer(&input_ptrs, &output_ptrs, stream.cu_stream(), batch_size)
            .unwrap();
    }

    info!("Running {} inference iterations...", args.iterations);
    let mut latencies = Vec::with_capacity(args.iterations);
    let mut total_time = Duration::ZERO;

    for i in 0..args.iterations {
        let start = Instant::now();
        engine
            .infer(&input_ptrs, &output_ptrs, stream.cu_stream(), batch_size)
            .unwrap();
        let elapsed = start.elapsed();
        latencies.push(elapsed);
        total_time += elapsed;

        let interval = (args.iterations / 10).max(1);
        if i % interval == 0 && i > 0 {
            let avg = total_time.as_secs_f32() / i as f32;
            info!(
                "Progress: {}/{} ({:.1}%), avg latency: {:.3}ms",
                i,
                args.iterations,
                (i as f32 / args.iterations as f32) * 100.0,
                avg * 1000.0,
            );
        }
    }

    let total_latency: f32 = latencies.iter().map(|t| t.as_secs_f32()).sum();
    let avg_batch_latency = total_latency / latencies.len() as f32;
    let avg_frame_latency = total_latency / (latencies.len() as f32 * batch_size as f32);

    info!("inference calls    : {}", args.iterations);
    info!("total latency      : {}", total_latency);
    info!("avg. frame latency : {}", avg_frame_latency);
    info!("avg. frame fps     : {}", 1.0 / avg_frame_latency);
    info!("avg. batch latency : {}", avg_batch_latency);
    info!("avg. batch fps     : {}", 1.0 / avg_batch_latency);
}
