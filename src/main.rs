//! Simple program to run tests and benchmark for libinfer.

use approx::assert_relative_eq;
use cxx::UniquePtr;
use libinfer::{Engine, Options, Precision};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    iter::{repeat, zip},
    path::PathBuf,
    str::FromStr,
    time::{Duration, Instant},
};

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
    let input_dim = engine.get_input_dims();
    assert_eq!(input_dim[0], 1);
    assert_eq!(input_dim[1], 3);
    assert_eq!(input_dim[2], 640);
    assert_eq!(input_dim[3], 640);
}

fn test_output_dim(engine: &UniquePtr<Engine>) {
    let output_dim = engine.get_output_dims();
    assert_eq!(output_dim[0], 1);
    assert_eq!(output_dim[1], 84);
    assert_eq!(output_dim[2], 8400);
}

fn test_output_features(engine: &mut UniquePtr<Engine>) {
    let batch_size = engine.get_input_dims()[0] - 1;
    let input = {
        let mut v = read_binary_f32("test/input.bin".into());
        if batch_size > 0 {
            let c = v.clone();
            v.extend(
                std::iter::repeat(&c)
                    .take(batch_size as usize)
                    .flat_map(|v| v.iter().cloned()),
            );
        }
        v
    };
    let first_twelve_expected = parse_file_to_float_vec("test/features.txt".into());

    let expected_output_size = engine
        .get_output_dims()
        .iter()
        .fold(1, |acc, &e| acc * e as usize);
    let batch_element_size = engine
        .get_output_dims()
        .iter()
        .skip(1)
        .fold(1, |acc, &e| acc * e as usize);

    let actual = engine.pin_mut().infer(&input).unwrap();

    // Check that the entire output length is correct.
    assert_eq!(actual.len(), expected_output_size);

    // Only checking the first twelve produced values. Repeat for each batch element.
    actual.chunks_exact(batch_element_size).for_each(|chunk| {
        zip(chunk, first_twelve_expected.clone()).for_each(|(a, e)| {
            assert_relative_eq!(*a, e, epsilon = 0.001);
        });
    });
}

fn benchmark_inference(engine: &mut UniquePtr<Engine>, num_runs: u64) {
    let input_dim = engine.get_input_dims();
    let batch_size = input_dim[0];
    let input_len = input_dim.iter().fold(1, |acc, &e| acc * e) as usize;
    let input_data: Vec<f32> = repeat(0.0).take(input_len).collect();

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

/// Benchmark inference engine.
fn main() {
    let n = 2 << 15;
    let b1_options = Options {
        model_name: "yolov8n_b1".into(),
        search_path: "test".into(),
        save_path: "test".into(),
        device_index: 0,
        precision: Precision::FP16,
        optimized_batch_size: 1,
        max_batch_size: 1,
    };
    let mut b1_engine = Engine::new(&b1_options).unwrap();

    test_input_dim(&b1_engine);
    test_output_dim(&b1_engine);

    let dynamic_options = Options {
        model_name: "clip".into(),
        search_path: "test".into(),
        save_path: "test".into(),
        device_index: 0,
        precision: Precision::FP16,
        optimized_batch_size: 32,
        max_batch_size: 128,
    };
    let dynamic_engine = Engine::new(&dynamic_options).unwrap();

    let b2_options = Options {
        model_name: "yolov8n_b2".into(),
        optimized_batch_size: 2,
        max_batch_size: 2,
        ..b1_options.clone()
    };
    let mut b2_engine = Engine::new(&b2_options).unwrap();

    let b4_options = Options {
        model_name: "yolov8n_b4".into(),
        optimized_batch_size: 4,
        max_batch_size: 4,
        ..b1_options.clone()
    };
    let mut b4_engine = Engine::new(&b4_options).unwrap();

    let b8_options = Options {
        model_name: "yolov8n_b8".into(),
        optimized_batch_size: 8,
        max_batch_size: 8,
        ..b1_options.clone()
    };
    let mut b8_engine = Engine::new(&b8_options).unwrap();

    let b16_options = Options {
        model_name: "yolov8n_b16".into(),
        optimized_batch_size: 16,
        max_batch_size: 16,
        ..b1_options.clone()
    };
    let mut b16_engine = Engine::new(&b16_options).unwrap();

    test_output_features(&mut b1_engine);
    test_output_features(&mut b4_engine);

    benchmark_inference(&mut b1_engine, n);
    benchmark_inference(&mut b2_engine, n / 2);
    benchmark_inference(&mut b4_engine, n / 4);
    benchmark_inference(&mut b8_engine, n / 8);
    benchmark_inference(&mut b16_engine, n / 16);
}
