# `libinfer`
This library provides a simple Rust interface to a TensorRT engine using [cxx](https://cxx.rs/)

## Overview
`libinfer` allows for seamless integration of TensorRT models into Rust applications with minimal overhead. The library handles the complex C++ interaction with TensorRT while exposing a simple, idiomatic Rust API.

## Installation
To use this library, you'll need:
- CUDA and TensorRT installed on your system
- Environment variables set properly:
  - `TENSORRT_LIBRARIES`: Path to TensorRT libraries
  - `CUDA_LIBRARIES`: Path to CUDA libraries
  - `CUDA_INCLUDE_DIRS`: Path to CUDA include directories

Add to your `Cargo.toml`:
```toml
[dependencies]
libinfer = "0.0.3"
```

## Usage
The goal of the API is to keep as much processing in Rust land as possible. Here is a sample usage:

```rust
let options = Options {
    path: "yolov8n.engine".into(),
    device_index: 0,
};
let mut engine = Engine::new(&options).unwrap();

// Get input dimensions of the engine as [Channels, Height, Width]
let dims = engine.get_input_dims();

// Construct a dummy input (uint8 or float32 depending on model)
let input_size = dims.iter().fold(1, |acc, &e| acc * e as usize);
let input = InputTensor {
    name: "input".to_string();
    data: vec![0u8; input_size];

// Run inference
let output = engine.pin_mut().infer(&input).unwrap();

// Postprocess the output according to your model's output format
// ...
```

This library is intended to be used with pre-built TensorRT engines created by the Python API or the `trtexec` CLI tool for the target device.

## Features
- Support for both fixed and dynamic batch sizes
- Automatic handling of different input data types (UINT8, FP32)
- Direct access to model dimensions and parameters
- Error handling via Rust's `Result` type
- Logging integration with `RUST_LOG` environment variable

## Examples
Check the `examples/` directory for working examples:
- `basic.rs`: Simple inference example
- `benchmark.rs`: Performance benchmarking with various batch sizes
- `dynamic.rs`: Working with dynamic batch sizes
- `functional_test.rs`: Testing correctness of model outputs

Run an example with:
```
cargo run --example basic -- --path /path/to/model.engine
```

### Example Requirements
- You must provide your own TensorRT engine files (.engine)
- For the functional_test example, you'll need input.bin and features.txt files
- To create engine files, use NVIDIA's TensorRT tools such as:
  - TensorRT Python API
  - trtexec command-line tool
  - ONNX -> TensorRT conversion tools

See the documentation in each example file for specific requirements.

## Current Limitations
- The underlying engine code is not threadsafe (and the Rust binding does not implement `Sync`)
- Engine instances are `Send` but not `Sync`
- Input and output data transfers happen on the CPU-GPU boundary

## Future Work
- Allow passing device pointers and CUDA streams for stream synchronization events
- Async execution support

## Credits
Much of the C++ code is based on the [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) repo.
