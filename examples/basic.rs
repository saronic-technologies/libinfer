//! Demonstrates basic functionality of libinfer by running inference
//! on a TensorRT engine with synthetic input.
//!
//! ## Usage
//! ```bash
//! cargo run --example basic -- --path /path/to/your/model.engine
//! ```

use clap::Parser;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use libinfer::{Engine, Options};
use std::path::PathBuf;
use std::process;
use tracing::{error, info, Level};
use tracing::subscriber::set_global_default;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
#[clap(about = "Basic example for libinfer")]
struct Args {
    #[arg(short, long, value_name = "PATH", value_parser)]
    path: PathBuf,

    #[arg(short, long, value_name = "ITERATIONS", default_value_t = 1 << 10)]
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
    let batch_size = batch_dims.opt;

    info!("Engine loaded successfully");
    info!("Batch dimensions: {:?}", batch_dims);
    for ti in &input_infos {
        info!("  Input '{}': {:?} {:?}", ti.name, ti.dims, ti.dtype);
    }
    for ti in &output_infos {
        info!("  Output '{}': {:?} {:?}", ti.name, ti.dims, ti.dtype);
    }

    let stream = ctx.new_stream().expect("failed to create stream");

    let input_bufs: Vec<_> = input_infos
        .iter()
        .map(|ti| {
            stream
                .alloc_zeros::<u8>(ti.byte_size() * batch_size as usize)
                .expect("input alloc failed")
        })
        .collect();

    let mut output_bufs: Vec<_> = output_infos
        .iter()
        .map(|ti| {
            stream
                .alloc_zeros::<u8>(ti.byte_size() * batch_size as usize)
                .expect("output alloc failed")
        })
        .collect();

    let input_refs: Vec<_> = input_bufs.iter().map(|b| b.device_ptr(&stream)).collect();
    let output_refs: Vec<_> = output_bufs
        .iter_mut()
        .map(|b| b.device_ptr_mut(&stream))
        .collect();
    let input_ptrs: Vec<_> = input_refs.iter().map(|(ptr, _)| *ptr).collect();
    let output_ptrs: Vec<_> = output_refs.iter().map(|(ptr, _)| *ptr).collect();

    info!("Running inference for {} iterations...", args.iterations);

    for i in 0..args.iterations {
        if i % (args.iterations / 10).max(1) == 0 {
            info!("Iteration {}/{}", i, args.iterations);
        }
        if let Err(e) = engine.infer(&input_ptrs, &output_ptrs, stream.cu_stream(), batch_size) {
            error!("Inference error: {e}");
            break;
        }
    }

    info!("Inference complete!");
}
