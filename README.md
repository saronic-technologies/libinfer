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
libinfer = "0.0.5"
```

## Usage
The goal of the API is to keep as much processing in Rust land as possible. Here is a sample usage:

```rust
use libinfer::{Engine, Options, InputTensor, TensorDataType};

let options = Options {
    path: "model.engine".into(),
    device_index: 0,
};
let mut engine = Engine::new(&options).unwrap();

// Query per-input shape profiles (includes batch dimension)
let profiles = engine.get_input_shape_profiles();
let input_infos = engine.get_input_dims(); // dims without batch

// Construct inputs using each input's optimal shape
let input_tensors: Vec<InputTensor> = profiles.iter().zip(input_infos.iter()).map(|(profile, info)| {
    let shape = &profile.opt_shape;
    let dtype_size = match info.dtype {
        TensorDataType::FP32 => 4,
        TensorDataType::UINT8 | TensorDataType::BOOL => 1,
        TensorDataType::INT64 => 8,
        _ => 1,
    };
    let byte_count: usize = shape.iter().map(|&d| d as usize).product::<usize>() * dtype_size;
    InputTensor {
        name: info.name.clone(),
        data: vec![0u8; byte_count],
        dtype: info.dtype.clone(),
    }
}).collect();

// Run inference
let outputs = engine.pin_mut().infer(&input_tensors).unwrap();

// Process outputs
for output in &outputs {
    println!("Output '{}': {} bytes", output.name, output.data.len());
}
```

This library is intended to be used with pre-built TensorRT engines created by the Python API or the `trtexec` CLI tool for the target device.

## Features
- Heterogeneous per-input dynamic shapes — each input tensor can have its own independent dynamic dimension
- Per-input shape profiles (`get_input_shape_profiles()`) exposing min/opt/max shapes from the TensorRT optimization profile
- Support for UINT8, FP32, INT64, and BOOL input data types
- Direct access to model dimensions and parameters
- Error handling via Rust's `Result` type
- Logging integration with `RUST_LOG` environment variable

## Examples
Check the `examples/` directory for working examples:
- `basic.rs`: Simple inference example
- `benchmark.rs`: Performance benchmarking with per-input shape profiling (tests min/opt/max automatically)
- `dynamic.rs`: Working with heterogeneous per-input dynamic shapes
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

## Synchronization Model

No `cudaStreamSynchronize` is needed between H2D copies and `enqueueV3`. This is safe for several reasons:

1. **Stream ordering** — all H2D copies and `enqueueV3` are submitted to the same CUDA stream, which guarantees in-order execution. Copies complete before kernels begin.
2. **Pageable host memory** — input data comes from Rust `Vec<u8>` on the regular heap (not pinned memory). `cudaMemcpyAsync` with pageable memory blocks the CPU until the copy is staged, making a subsequent sync redundant.
3. **TensorRT auxiliary streams** — TRT may use auxiliary streams internally during `enqueueV3`, but it inserts event synchronizations so all auxiliary work waits on the main stream at entry and the main stream waits on all auxiliary work at exit.
4. **CPU-side calls** — `setInputShape`, `allInputDimensionsSpecified`, and `setTensorAddress` are pure CPU operations with no stream interaction.

A post-inference `cudaStreamSynchronize` is still required to ensure D2H output copies are complete before reading results. `infer()` handles this internally.

## Current Limitations
- Each input tensor supports at most one dynamic dimension (batch). Multiple dynamic dims per input would require explicit shape specification.
- The underlying engine code is not threadsafe (and the Rust binding does not implement `Sync`)
- Engine instances are `Send` but not `Sync`
- Input and output data transfers happen on the CPU-GPU boundary

## Future Work
- Allow passing device pointers and CUDA streams for stream synchronization events
- Async execution support

## Credits
Much of the C++ code is based on the [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) repo.
