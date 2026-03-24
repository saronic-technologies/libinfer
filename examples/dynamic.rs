//! Demonstrates using a TensorRT engine with dynamic batch sizes.
//!
//! ## Usage
//! ```bash
//! cargo run --example dynamic -- --path /path/to/your/dynamic_batch.engine
//! ```

use clap::Parser;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use libinfer::{Engine, Options};
use std::path::PathBuf;
use std::process;
use std::time::Instant;
use tracing::{error, info, warn, Level};
use tracing::subscriber::set_global_default;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
#[clap(about = "Dynamic batch size example for libinfer")]
struct Args {
    #[arg(short, long, value_name = "PATH", value_parser)]
    path: PathBuf,

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
    info!("Loading TensorRT engine from: {}", args.path.display());

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
    let batch_dims = engine.get_batch_dims();

    info!("Engine loaded successfully");
    info!(
        "Batch dimensions: min={}, optimal={}, max={}",
        batch_dims.min, batch_dims.opt, batch_dims.max
    );

    if batch_dims.min == batch_dims.max {
        warn!("This engine does not support dynamic batch sizes!");
        warn!("All batch dimensions are fixed at {}", batch_dims.min);
        return;
    }

    let stream = ctx.new_stream().expect("failed to create stream");

    let batch_sizes = [batch_dims.min, batch_dims.opt, batch_dims.max];

    for &batch_size in &batch_sizes {
        info!("\nTesting batch size: {}", batch_size);

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

        for _ in 0..5 {
            let _ = engine.infer(&input_ptrs, &output_ptrs, stream.cu_stream(), batch_size);
        }

        let start = Instant::now();
        let result = engine.infer(&input_ptrs, &output_ptrs, stream.cu_stream(), batch_size);
        let elapsed = start.elapsed();

        match result {
            Ok(()) => {
                info!("Inference successful!");
                info!("Inference time: {:?}", elapsed);
                info!(
                    "Throughput: {:.2} items/second",
                    batch_size as f64 / elapsed.as_secs_f64()
                );
            }
            Err(e) => {
                error!("Inference failed: {}", e);
            }
        }
    }
}
