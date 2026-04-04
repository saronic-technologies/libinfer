# libinfer

Rust interface to TensorRT engines via [cxx](https://cxx.rs/). Caller provides device memory and CUDA streams.

## Installation

Requirements:
- CUDA and TensorRT installed
- Environment variables:
  - `TENSORRT_LIBRARIES`: path to TensorRT libraries
  - `CUDA_LIBRARIES`: path to CUDA libraries
  - `CUDA_INCLUDE_DIRS`: path to CUDA include directories

```toml
[dependencies]
libinfer = "0.1.0"
```

A Nix flake is provided for development. `nix develop` sets up all dependencies.

## Usage

The API operates on raw CUDA device pointers and streams. The caller is responsible for
device selection, memory allocation, and stream management.

```rust
use cudarc::driver::CudaContext;
use libinfer::{Engine, Options};

// Set the CUDA device before loading the engine
let ctx = CudaContext::new(0).expect("failed to create CUDA context");

let options = Options {
    path: "model.engine".into(),
};
let mut engine = Engine::new(&options).unwrap();

// Query tensor metadata
let inputs = engine.get_input_dims();
let outputs = engine.get_output_dims();
let batch = engine.get_batch_dims();

// Allocate device memory, run inference
let stream = ctx.new_stream().unwrap();
// ... allocate input_bufs, output_bufs on the device ...

engine.infer(&input_ptrs, &output_ptrs, stream.cu_stream(), batch.opt).unwrap();
```

Input and output pointer arrays must match the order returned by `get_input_dims()` / `get_output_dims()`.

## Examples

```
cargo run --example basic -- --path /path/to/model.engine
cargo run --example benchmark -- --path /path/to/model.engine
cargo run --example dynamic -- --path /path/to/model.engine
```

## Testing

Tests require a CUDA-capable GPU. Generate test models and build TensorRT engines:

```bash
python3 test/generate_models.py

trtexec --onnx=test/test_dynamic.onnx --saveEngine=test/test_dynamic.engine \
    --minShapes=input:1x4 --optShapes=input:4x4 --maxShapes=input:8x4

trtexec --onnx=test/test_multi_input.onnx --saveEngine=test/test_multi_input.engine \
    --minShapes=input_a:1x3,input_b:1x5 --optShapes=input_a:4x3,input_b:4x5 \
    --maxShapes=input_a:8x3,input_b:8x5
```

Then run:

```
cargo test
```

## Caveats

- `Engine` is `Send` but not `Sync`. `infer` takes `&mut self`. For concurrent inference on the same model, create separate `Engine` instances.
- The caller must ensure the CUDA context outlives the engine, particularly when cudarc's event tracking is disabled.
- Only the batch dimension is dynamic. Non-batch dynamic shapes are yet not supported.
- Engine files are not portable across TensorRT versions or GPU architectures. Rebuild from ONNX for each target.
- CUDA graphs are not yet supported.

## Credits

C++ code originally based on [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api).
