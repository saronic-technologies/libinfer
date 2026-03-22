//! Validates correct functionality of a TensorRT engine by checking
//! output values against expected reference values.
//!
//! ## Usage
//! ```bash
//! cargo run --example functional_test -- --path /path/to/test/directory
//! ```
//!
//! Required test files: yolov8n.engine, input.bin, features.txt

use approx::assert_relative_eq;
use clap::Parser;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use libinfer::{Engine, Options};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::iter::zip;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;
use tracing::{error, info, Level};
use tracing::subscriber::set_global_default;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, value_name = "PATH", default_value = ".", value_parser)]
    path: PathBuf,
}

fn main() {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(Level::INFO)
        .finish();
    set_global_default(subscriber).expect("Failed to set tracing subscriber");

    let args = Args::parse();

    let engine_path = if args.path.is_file() {
        args.path.clone()
    } else {
        args.path.join("yolov8n.engine")
    };

    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    unsafe { ctx.disable_event_tracking() };

    let options = Options {
        path: engine_path.to_string_lossy().to_string(),
    };

    let mut engine = Engine::new(&options, &ctx).unwrap_or_else(|e| {
        error!("Failed to load engine: {e}");
        process::exit(1);
    });

    let input_infos = engine.get_input_dims();
    let output_infos = engine.get_output_dims();
    let batch_dims = engine.get_batch_dims();
    let batch_size = batch_dims.opt;

    assert_eq!(input_infos.len(), 1);
    assert_eq!(input_infos[0].dims, &[3, 640, 640]);
    info!("Input: '{}' {:?}", input_infos[0].name, input_infos[0].dims);

    assert_eq!(output_infos.len(), 1);
    assert_eq!(output_infos[0].dims, &[84, 8400]);
    info!(
        "Output: '{}' {:?}",
        output_infos[0].name, output_infos[0].dims
    );

    assert_eq!(batch_dims.min, 1);
    assert_eq!(batch_dims.opt, 1);
    assert_eq!(batch_dims.max, 1);

    let host_input = {
        let mut file = File::open(args.path.join("input.bin")).expect("failed to open input.bin");
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .expect("failed to read input.bin");
        buf
    };

    let expected: Vec<f32> = {
        let file = File::open(args.path.join("features.txt")).expect("failed to open features.txt");
        BufReader::new(file)
            .lines()
            .flat_map(|line| {
                line.unwrap()
                    .split_whitespace()
                    .filter_map(|s| f32::from_str(s).ok())
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    let stream = ctx.new_stream().expect("failed to create stream");

    let input_buf = stream.clone_htod(&host_input).expect("H2D failed");
    let mut output_buf = stream
        .alloc_zeros::<u8>(output_infos[0].byte_size() * batch_size as usize)
        .expect("alloc failed");

    {
        let (input_ptr, _ig) = input_buf.device_ptr(&stream);
        let (output_ptr, _og) = output_buf.device_ptr_mut(&stream);

        engine
            .infer(&[input_ptr], &[output_ptr], stream.cu_stream(), batch_size)
            .expect("inference failed");
    }

    let host_output: Vec<u8> = stream.clone_dtoh(&output_buf).expect("D2H failed");

    let actual_f32: Vec<f32> = host_output
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let output_elems = output_infos[0].elem_count();
    assert_eq!(actual_f32.len(), output_elems * batch_size as usize);

    actual_f32.chunks_exact(output_elems).for_each(|chunk| {
        zip(chunk, &expected).for_each(|(a, e)| {
            assert_relative_eq!(*a, e, epsilon = 0.1);
        });
    });

    info!("Output features agree");
}
