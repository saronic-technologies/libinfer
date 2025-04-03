//! Demonstrates basic functionality of `libinfer`.

use clap::Parser;
use libinfer::{Engine, Options};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the engine file.
    #[arg(short, long, value_name = "PATH", value_parser)]
    path: PathBuf,

    /// Number of iterations to run.
    #[arg(short, long, value_name = "ITERATIONS", default_value_t = 1 << 19)]
    iterations: usize,
}

fn main() {
    let args = Args::parse();

    let options = Options {
        path: args
            .path
            .to_string_lossy()
            .to_owned()
            .to_string(),
        device_index: 0,
    };
    let mut engine = Engine::new(&options).unwrap();

    let input_size = engine
        .get_input_dims()
        .iter()
        .fold(1, |acc, &e| acc * e as usize);

    let input = vec![0; input_size];

    for _ in 0..args.iterations {
        engine.pin_mut().infer(&input).inspect_err(|e| println!("error: {e}"));
    }
}
