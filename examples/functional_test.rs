//! # Functional Test Example
//!
//! Validates the correct functionality of a TensorRT engine by checking:
//! - Input dimensions
//! - Output dimensions
//! - Batch dimensions
//! - Output values against expected reference values
//!
//! ## Usage
//! ```bash
//! cargo run --example functional_test -- --path /path/to/test/directory
//! ```
//!
//! ## Required Test Files
//! This example requires the following files in your test directory:
//! - `yolov8n.engine`: The TensorRT engine file to test (or specify direct path to engine)
//! - `input.bin`: A binary file containing raw input data matching the model's input format
//! - `features.txt`: A text file with expected output values (space-separated floats)
//!
//! ## Note
//! You must provide your own engine file and corresponding test data. The test assumes
//! a YOLOv8n model by default, but the code can be adapted for any model architecture.

use anyhow::Result;
use approx::assert_relative_eq;
use clap::Parser;
use cxx::UniquePtr;
use libinfer::{Engine, Options};
use libinfer::ffi::InputTensor;
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
    iter::{repeat, zip},
    path::PathBuf,
    str::FromStr,
};
use tracing::{info, error, Level};
use tracing_subscriber::{FmtSubscriber, EnvFilter};

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
    let input_dims = engine.get_input_dims();
    assert_eq!(input_dims.len(), 1); // Expecting one input tensor
    let input_dim = &input_dims[0];
    assert_eq!(input_dim.dims[0], 3);
    assert_eq!(input_dim.dims[1], 640);
    assert_eq!(input_dim.dims[2], 640);
    info!("Input dimensions: '{}' -> {:?}", input_dim.name, input_dim.dims);
}

fn test_batch_dim(engine: &UniquePtr<Engine>) {
    let batch_dim = engine.get_batch_dims();
    assert_eq!(batch_dim.min, 1);
    assert_eq!(batch_dim.opt, 1);
    assert_eq!(batch_dim.max, 1);
    info!("Batch dimensions: {batch_dim:?}");
}

fn test_output_dim(engine: &UniquePtr<Engine>) {
    let output_dims = engine.get_output_dims();
    assert_eq!(output_dims.len(), 1); // Expecting one output tensor
    let output_dim = &output_dims[0];
    assert_eq!(output_dim.dims[0], 84);
    assert_eq!(output_dim.dims[1], 8400);
    info!("Output dimensions: '{}' -> {:?}", output_dim.name, output_dim.dims);
}

fn test_output_features(engine: &mut UniquePtr<Engine>, input: &[u8], expected: &[f32]) {
    info!("Testing output features...");
    let batch_size = engine.get_batch_dims().opt;
    let input_names = engine.get_input_names();
    
    // Create TensorInput for the first input tensor
    let ext_input_data = {
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

    let input_tensors = vec![InputTensor {
        name: input_names[0].clone(),
        data: ext_input_data,
    }];

    let output_dims = engine.get_output_dims();
    let expected_output_size = output_dims[0]
        .dims
        .iter()
        .fold(1, |acc, &e| acc * e as usize);
    let batch_element_size = output_dims[0]
        .dims
        .iter()
        .fold(1, |acc, &e| acc * e as usize);

    let output_tensors = engine.pin_mut().infer(&input_tensors).unwrap();
    
    // Get the first output tensor
    let actual = &output_tensors[0].data;

    // Check that the entire output length is correct.
    assert_eq!(actual.len(), expected_output_size);

    // Only checking the first twelve produced values. Repeat for each batch element.
    actual.chunks_exact(batch_element_size).for_each(|chunk| {
        zip(chunk, expected).for_each(|(a, e)| {
            assert_relative_eq!(*a, e, epsilon = 0.1);
        });
    });

    info!("Output features agree");
}

fn main() {
    // Initialize tracing subscriber for logging
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");

    let args = Args::parse();

    // Check if path is directly to an engine file
    let engine_path = if args.path.is_file() {
        args.path.clone()
    } else {
        args.path.join("yolov8n.engine")
    };

    let options = Options {
        path: engine_path.to_string_lossy().to_string(),
        device_index: 0,
    };

    let mut engine = match Engine::new(&options) {
        Ok(engine) => engine,
        Err(e) => {
            error!("Failed to load engine: {e}");
            std::process::exit(1);
        }
    };

    let input = read_binary_file(args.path.join("input.bin")).unwrap_or_else(|e| {
        error!("Failed to read input.bin: {e}");
        std::process::exit(1);
    });
    let expected = parse_file_to_float_vec(args.path.join("features.txt")).unwrap_or_else(|e| {
        error!("Failed to parse features.txt: {e}");
        std::process::exit(1);
    });

    info!("Input data type: {:?}", engine.get_input_data_type());

    test_input_dim(&engine);
    test_output_dim(&engine);
    test_batch_dim(&engine);
    test_output_features(&mut engine, &input, &expected);
}
