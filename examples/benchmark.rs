//! Simple benchmark for `libinfer` over a range of batch sizes.

//use approx::assert_relative_eq;
use clap::Parser;
use cxx::UniquePtr;
use libinfer::{Engine, InputDataType, Options};
use std::{
    iter::repeat,
    path::PathBuf,
    time::{Duration, Instant},
};

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
    let input_dim = engine.get_input_dims();
    let batch_size = input_dim[0];
    let input_len = input_dim.iter().fold(1, |acc, &e| acc * e) as usize;
    let dtype = engine.get_input_data_type();
    let input_data: Vec<u8> = match dtype {
        InputDataType::UINT8 => repeat(0).take(input_len).collect(),
        InputDataType::FP32 => repeat(0).take(4 * input_len).collect(),
        _ => unreachable!(),
    };

    // Warmup.
    println!("Warming up inference codepath...");
    for _ in 0..1024 {
        let _output = engine.pin_mut().infer(&input_data).unwrap();
    }

    // Measure.
    println!("Beginning {num_runs} inference runs...");
    let latencies = (0..num_runs)
        .map(|_| {
            let start = Instant::now();
            let _output = engine.pin_mut().infer(&input_data).unwrap();
            start.elapsed()
        })
        .collect::<Vec<Duration>>();

    let total_latency = latencies.iter().map(|t| t.as_secs_f32()).sum::<f32>();
    let average_batch_latency = total_latency / latencies.len() as f32;
    let average_batch_framerate = 1.0 / average_batch_latency;
    let average_frame_latency = total_latency / (latencies.len() as f32 * batch_size as f32);
    let average_frame_framerate = 1.0 / average_frame_latency;

    println!("inference calls    : {}", num_runs);
    println!("total latency      : {}", total_latency);
    println!("avg. frame latency : {}", average_frame_latency);
    println!("avg. frame fps     : {}", average_frame_framerate);
    println!("avg. batch latency : {}", average_batch_latency);
    println!("avg. batch fps     : {}", average_batch_framerate);
}

fn main() {
    let args = Args::parse();

    let b1_options = Options {
        path: args
            .path
            .join("yolov8n.engine")
            .to_string_lossy()
            .to_owned()
            .to_string(),
        device_index: 0,
    };
    let mut b1_engine = Engine::new(&b1_options).unwrap();

    println!("Input data type: {:?}", b1_engine.get_input_data_type());

    let b2_options = Options {
        path: args
            .path
            .join("yolov8n_b2.engine")
            .to_string_lossy()
            .to_owned()
            .to_string(),
        device_index: 0,
    };
    let mut b2_engine = Engine::new(&b2_options).unwrap();

    let b4_options = Options {
        path: args
            .path
            .join("yolov8n_b4.engine")
            .to_string_lossy()
            .to_owned()
            .to_string(),
        device_index: 0,
    };
    let mut b4_engine = Engine::new(&b4_options).unwrap();

    let b8_options = Options {
        path: args
            .path
            .join("yolov8n_b8.engine")
            .to_string_lossy()
            .to_owned()
            .to_string(),
        device_index: 0,
    };
    let mut b8_engine = Engine::new(&b8_options).unwrap();

    let b16_options = Options {
        path: args
            .path
            .join("yolov8n_b16.engine")
            .to_string_lossy()
            .to_owned()
            .to_string(),
        device_index: 0,
    };
    let mut b16_engine = Engine::new(&b16_options).unwrap();

    benchmark_inference(&mut b1_engine, args.iterations);
    benchmark_inference(&mut b2_engine, args.iterations / 2);
    benchmark_inference(&mut b4_engine, args.iterations / 4);
    benchmark_inference(&mut b8_engine, args.iterations / 8);
    benchmark_inference(&mut b16_engine, args.iterations / 16);
}
