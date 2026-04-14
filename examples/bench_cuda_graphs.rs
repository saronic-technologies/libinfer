//! Benchmarks CUDA graph capture/replay vs. plain enqueueV3.
//!
//! Allocates N distinct input/output buffer sets and cycles through them.
//! After warmup (which captures all graphs), the timed loop exercises the
//! cache with rotating pointer sets.
//!
//! ## Usage
//! ```bash
//! # First generate and build the benchmark model:
//! #   python test/generate_models.py
//! #   trtexec --onnx=test/bench_deep_mlp.onnx --saveEngine=test/bench_deep_mlp.engine \
//! #           --minShapes=input:1x512 --optShapes=input:16x512 --maxShapes=input:64x512
//!
//! cargo run --release --example bench_cuda_graphs -- --path test/bench_deep_mlp.engine
//! ```

use clap::Parser;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use cudarc::driver::sys::CUstream;
use libinfer::{Engine, Options};
use std::path::PathBuf;
use std::process;
use std::time::{Duration, Instant};
use tracing::{error, info, Level};
use tracing::subscriber::set_global_default;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
#[clap(about = "Benchmark CUDA graphs vs plain enqueueV3")]
struct Args {
    #[arg(short, long)]
    path: PathBuf,

    #[arg(short, long, default_value_t = 1 << 14)]
    iterations: usize,

    #[arg(short, long, default_value_t = 1024)]
    warmup: usize,

    #[arg(short, long, default_value_t = 0)]
    device: u32,

    /// Batch size to benchmark. Defaults to the engine's optimal batch size.
    #[arg(short, long)]
    batch_size: Option<u32>,

    /// Number of distinct input/output buffer sets to cycle through.
    #[arg(short, long, default_value_t = 10)]
    num_slots: usize,
}

struct BenchResult {
    total: Duration,
    avg_us: f64,
    p50_us: f64,
    p99_us: f64,
    min_us: f64,
}

fn run_bench(
    engine: &mut Engine,
    slots: &[(&[u64], &[u64])],
    stream: CUstream,
    batch_size: u32,
    warmup: usize,
    iterations: usize,
) -> BenchResult {
    let n = slots.len();

    for i in 0..warmup {
        let (ip, op) = slots[i % n];
        engine
            .infer(ip, op, stream, batch_size)
            .expect("warmup failed");
    }

    let mut latencies = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let (ip, op) = slots[i % n];
        let start = Instant::now();
        engine
            .infer(ip, op, stream, batch_size)
            .expect("inference failed");
        latencies.push(start.elapsed());
    }

    latencies.sort();
    let total: Duration = latencies.iter().sum();
    let to_us = |d: Duration| d.as_secs_f64() * 1e6;

    BenchResult {
        total,
        avg_us: to_us(total) / iterations as f64,
        p50_us: to_us(latencies[iterations / 2]),
        p99_us: to_us(latencies[iterations * 99 / 100]),
        min_us: to_us(latencies[0]),
    }
}

fn print_table(label: &str, iterations: usize, batch_size: u32, num_slots: usize, plain: &BenchResult, graph: &BenchResult) {
    let speedup = plain.avg_us / graph.avg_us;

    println!();
    println!("=== {} ({} iters, batch={}, {} slots) ===", label, iterations, batch_size, num_slots);
    println!("{:<20} {:>12} {:>12}", "", "enqueueV3", "CUDA graph");
    println!("{:<20} {:>12} {:>12}", "─".repeat(20), "─".repeat(12), "─".repeat(12));
    println!("{:<20} {:>10.1}us {:>10.1}us", "avg latency", plain.avg_us, graph.avg_us);
    println!("{:<20} {:>10.1}us {:>10.1}us", "p50 latency", plain.p50_us, graph.p50_us);
    println!("{:<20} {:>10.1}us {:>10.1}us", "p99 latency", plain.p99_us, graph.p99_us);
    println!("{:<20} {:>10.1}us {:>10.1}us", "min latency", plain.min_us, graph.min_us);
    println!("{:<20} {:>10.3}s  {:>10.3}s", "total", plain.total.as_secs_f64(), graph.total.as_secs_f64());
    println!();
    println!("Speedup: {:.2}x", speedup);
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

    let ctx = CudaContext::new(args.device as usize).expect("failed to create CUDA context");
    unsafe { ctx.disable_event_tracking() };

    let path_str = args.path.to_string_lossy().to_string();

    let mut engine_plain = Engine::new(&Options::new(path_str.clone(), None))
        .expect("failed to load engine (plain)");
    let mut engine_graph = Engine::new(&Options::new(path_str, Some(0)))
        .expect("failed to load engine (graph)");

    let input_infos = engine_plain.get_input_dims();
    let output_infos = engine_plain.get_output_dims();
    let batch_dims = engine_plain.get_batch_dims();
    let batch_size = args.batch_size.unwrap_or(batch_dims.opt);

    info!(
        "Engine: {} inputs, {} outputs, batch {}/{}/{}",
        input_infos.len(), output_infos.len(),
        batch_dims.min, batch_dims.opt, batch_dims.max,
    );
    info!(
        "Benchmark: batch_size={}, warmup={}, iterations={}, slots={}",
        batch_size, args.warmup, args.iterations, args.num_slots,
    );

    let stream = ctx.new_stream().expect("failed to create stream");

    // Allocate N distinct buffer sets
    let mut input_buf_sets = Vec::new();
    let mut output_buf_sets = Vec::new();

    for _ in 0..args.num_slots {
        let inputs: Vec<_> = input_infos
            .iter()
            .map(|ti| {
                stream
                    .alloc_zeros::<u8>(ti.byte_size() * batch_size as usize)
                    .expect("alloc failed")
            })
            .collect();
        let outputs: Vec<_> = output_infos
            .iter()
            .map(|ti| {
                stream
                    .alloc_zeros::<u8>(ti.byte_size() * batch_size as usize)
                    .expect("alloc failed")
            })
            .collect();
        input_buf_sets.push(inputs);
        output_buf_sets.push(outputs);
    }

    // Extract device pointers. We need stable references so we collect
    // the (ptr, token) pairs into a Vec that lives for the benchmark.
    let input_ptr_refs: Vec<Vec<_>> = input_buf_sets
        .iter()
        .map(|bufs| bufs.iter().map(|b| b.device_ptr(&stream)).collect())
        .collect();
    let output_ptr_refs: Vec<Vec<_>> = output_buf_sets
        .iter_mut()
        .map(|bufs| bufs.iter_mut().map(|b| b.device_ptr_mut(&stream)).collect())
        .collect();

    let input_ptrs: Vec<Vec<u64>> = input_ptr_refs
        .iter()
        .map(|refs| refs.iter().map(|(ptr, _)| *ptr).collect())
        .collect();
    let output_ptrs: Vec<Vec<u64>> = output_ptr_refs
        .iter()
        .map(|refs| refs.iter().map(|(ptr, _)| *ptr).collect())
        .collect();

    let slots: Vec<(&[u64], &[u64])> = input_ptrs
        .iter()
        .zip(output_ptrs.iter())
        .map(|(ip, op)| (ip.as_slice(), op.as_slice()))
        .collect();

    let cu_stream = stream.cu_stream();

    info!("Running plain enqueueV3...");
    let plain = run_bench(
        &mut engine_plain, &slots, cu_stream,
        batch_size, args.warmup, args.iterations,
    );

    info!("Running CUDA graph replay...");
    let graph = run_bench(
        &mut engine_graph, &slots, cu_stream,
        batch_size, args.warmup, args.iterations,
    );

    print_table("Results", args.iterations, batch_size, args.num_slots, &plain, &graph);
}
