//! Demonstrates basic functionality of `libinfer`.

use anyhow::Result;
use approx::assert_relative_eq;
use clap::Parser;
use cxx::UniquePtr;
use libinfer::{Engine, Options};
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    iter::{repeat, zip},
    path::PathBuf,
    str::FromStr,
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

fn read_binary_file(path: PathBuf) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

fn parse_file_to_float_vec(path: PathBuf) -> Result<Vec<f32>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut float_vec = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let values: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| f32::from_str(s).ok())
            .collect();

        float_vec.extend(values);
    }
    Ok(float_vec)
}

fn test_input_dim(engine: &UniquePtr<Engine>) {
    let input_dim = engine.get_input_dims();
    assert_eq!(input_dim[0], 3);
    assert_eq!(input_dim[1], 640);
    assert_eq!(input_dim[2], 640);
    println!("Input dimensions: {input_dim:?}");
}

fn test_batch_dim(engine: &UniquePtr<Engine>) {
    let batch_dim = engine.get_batch_dims();
    assert_eq!(batch_dim.min, 1);
    assert_eq!(batch_dim.opt, 1);
    assert_eq!(batch_dim.max, 1);
    println!("Batch dimensions: {batch_dim:?}");
}

fn test_output_dim(engine: &UniquePtr<Engine>) {
    let output_dim = engine.get_output_dims();
    assert_eq!(output_dim[0], 84);
    assert_eq!(output_dim[1], 8400);
    println!("Output dimensions: {output_dim:?}");
}

fn test_output_features(engine: &mut UniquePtr<Engine>, input: &[u8], expected: &[f32]) {
    println!("Testing output features...");
    let batch_size = engine.get_batch_dims().opt;
    let ext_input = {
        let mut v = input.to_vec();
        if batch_size > 1 {
            v.extend(
                repeat(input)
                    .take(batch_size as usize)
                    .flat_map(|v| v.iter().cloned()),
            );
        }
        v
    };

    let expected_output_size = engine
        .get_output_dims()
        .iter()
        .fold(1, |acc, &e| acc * e as usize);
    let batch_element_size = engine
        .get_output_dims()
        .iter()
        .fold(1, |acc, &e| acc * e as usize);

    let actual = engine.pin_mut().infer(&ext_input).unwrap();

    // Check that the entire output length is correct.
    assert_eq!(actual.len(), expected_output_size);

    // Only checking the first twelve produced values. Repeat for each batch element.
    actual.chunks_exact(batch_element_size).for_each(|chunk| {
        zip(chunk, expected).for_each(|(a, e)| {
            assert_relative_eq!(*a, e, epsilon = 0.1);
        });
    });

    println!("Output features agree");
}

fn main() {
    let args = Args::parse();

    let options = Options {
        path: args
            .path
            .join("yolov8n.engine")
            .to_string_lossy()
            .to_owned()
            .to_string(),
        device_index: 0,
    };
    let mut engine = Engine::new(&options).unwrap();

    let input = read_binary_file(args.path.join("input.bin")).unwrap();
    let expected = parse_file_to_float_vec(args.path.join("features.txt")).unwrap();

    println!("Input data type: {:?}", engine.get_input_data_type());

    test_input_dim(&engine);
    test_output_dim(&engine);
    test_batch_dim(&engine);
    test_output_features(&mut engine, &input, &expected);
}
