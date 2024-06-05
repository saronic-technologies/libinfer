//! Simple program to run tests and benchmark for libinfer.
//! Why is this a separate program and not a `cargo test`?
//! Cargo shits the bed and fails to link correctly against the `tests` build target for some
//! reason and I couldn't be bothered to figure out why. Probably some `build.rs` shit.

use approx::relative_eq;
use cxx::UniquePtr;
use libinfer::ffi::{get_input_dim, get_output_dim, make_engine, run_inference, Engine, Options, Precision};
use std::fs::File;
use std::io::Read;
use std::io::{BufRead, BufReader};
use std::iter::{repeat, zip};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Duration, Instant};

fn read_binary_f32(path: PathBuf) -> Vec<f32> {
    let mut f = File::open(path).unwrap();
    let mut input = Vec::new();
    f.read_to_end(&mut input).unwrap();
    let floats: Vec<f32> = input
        .chunks_exact(4)
        .map(|bs| f32::from_le_bytes(bs.try_into().unwrap()))
        .collect();
    floats
}

fn parse_file_to_float_vec(path: PathBuf) -> Vec<f32> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let mut float_vec = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let values: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| f32::from_str(s).ok())
            .collect();

        float_vec.extend(values);
    }
    float_vec
}

fn test_input_dim(engine: &UniquePtr<Engine>) {
    let input_dim = get_input_dim(&engine);
    assert_eq!(input_dim[0], 1);
    assert_eq!(input_dim[1], 3);
    assert_eq!(input_dim[2], 640);
    assert_eq!(input_dim[3], 640);
}

fn test_output_dim(engine: &UniquePtr<Engine>) {
    let output_dim = get_output_dim(&engine);
    assert_eq!(output_dim[0], 1);
    assert_eq!(output_dim[1], 84);
    assert_eq!(output_dim[2], 8400);
}

fn test_output_features(engine: &UniquePtr<Engine>) {
    let input = read_binary_f32("test/input.bin".into());
    let actual = run_inference(&engine, &input).unwrap();
    let expected = parse_file_to_float_vec("test/features.txt".into());
    let _ = zip(actual, expected).map(|(a, e)| relative_eq!(a, e, epsilon = 0.001));
}

fn benchmark_inference(engine: &UniquePtr<Engine>) {
    let input_dim = get_input_dim(&engine);
    let input_len = input_dim.iter().fold(1, |acc, &e| acc * e) as usize;
    let input_data: Vec<f32> = repeat(0.0).take(input_len).collect();

    // Warmup.
    println!("Warming up inference codepath...");
    for _ in 0..1024 {
        let _output = run_inference(&engine, &input_data).unwrap();
    }

    // Measure.
    println!("Testing inference...");
    let latencies = (0..1 << 12)
        .map(|_| {
            let start = Instant::now();
            let _output = run_inference(&engine, &input_data).unwrap();
            start.elapsed()
        })
        .collect::<Vec<Duration>>();

    let average_latency =
        latencies.iter().map(|t| t.as_secs_f32()).sum::<f32>() / latencies.len() as f32;
    let average_framerate = 1.0 / average_latency;

    println!("inference calls: {}", 2 << 15);
    println!("avg. latency   : {}", average_latency);
    println!("avg. fps       : {}", average_framerate);
}

/// Benchmark inference engine.
fn main() {
    let options = Options {
        model_name: "yolov8n_b1".into(),
        search_path: "test".into(),
        save_path: "test".into(),
        device_index: 0,
        precision: Precision::FP16,
        optimized_batch_size: 1,
        max_batch_size: 1
    };
    let engine = make_engine(&options).unwrap();

    test_input_dim(&engine);
    test_output_dim(&engine);
    test_output_features(&engine);
    benchmark_inference(&engine);

    let b2_options = Options {
        model_name: "yolov8n_b2".into(),
        optimized_batch_size: 2,
        max_batch_size: 2,
        ..options.clone()
    };
    let b2_engine = make_engine(&b2_options).unwrap();
    benchmark_inference(&b2_engine);

    let b4_options = Options {
        model_name: "yolov8n_b4".into(),
        optimized_batch_size: 4,
        max_batch_size: 4,
        ..options.clone()
    };
    let b4_engine = make_engine(&b4_options).unwrap();
    benchmark_inference(&b4_engine);

    let b8_options = Options {
        model_name: "yolov8n_b8".into(),
        optimized_batch_size: 8,
        max_batch_size: 8,
        ..options.clone()
    };
    let b8_engine = make_engine(&b8_options).unwrap();
    benchmark_inference(&b8_engine);


    let b16_options = Options {
        model_name: "yolov8n_b16".into(),
        optimized_batch_size: 16,
        max_batch_size: 16,
        ..options.clone()
    };
    let b16_engine = make_engine(&b16_options).unwrap();
    benchmark_inference(&b16_engine);
}
