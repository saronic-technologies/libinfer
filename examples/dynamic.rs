//! Demonstration of dynamic batch sizes.

use clap::Parser;
use libinfer::{Engine, Options};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the directory containing engine files.
    #[arg(short, long, value_name = "PATH", default_value = ".", value_parser)]
    path: PathBuf,

    /// Number of iterations (default: 32768)
    #[arg(short, long, value_name = "ITERATIONS", default_value_t = 1 << 15)]
    iterations: usize,
}

fn main() {
    let args = Args::parse();

    // A CLIP model which has been build using TensorRT to have a range of acceptable batch sizes.
    let options = Options {
        path: args
            .path
            .join("clip.engine")
            .to_string_lossy()
            .to_owned()
            .to_string(),
        device_index: 0,
    };
    let engine = Engine::new(&options).unwrap();

    println!("Input data type: {:?}", engine.get_input_data_type());
    println!("Input dimensions: {:?}", engine.get_input_dims());
    println!("Input batch dimensions: {:?}", engine.get_batch_dims());
}
