# `libinfer`
This library provides a simple Rust interface to a TensorRT engine using [cxx]([url](https://cxx.rs/))

## Usage
The goal of the API is to keep as much processing in Rust land as possible. Here is a sample usage.

```
let options = Options {
    path: "yolov8n.engine".into(),
    device_index: 0,
};
let mut engine = Engine::new(&options).unwrap();

// Get input dimensions of the engine as [Channels, Height, Width].
// For now, only a batch size of 1 is supported.
let dims = engine.get_input_dim();

// Construct a dummy input.
let input = vec![0.0; dims[0] * dims[1] * dims[2]];

// Run inference.
let output = engine.pin_mut().infer(&engine).unwrap();

// Postprocess the output according to your model's output format.
...
```

This library is intended to be used with pre-built TensorRT engines created by the Python API or the `trtexec` CLI tool for the target device.

Most errors are surfaced back to Rust via `Result`s. `libinfer` also reads the `RUST_LOG` environment variable and translates it to TensorRT's logger. The default log level is `warn`.


## Current Limitations
- The underlying engine code is not threadsafe (and the Rust binding does not implement `Sync`)
- Additional benchmarking and testing is desirable
- Dynamic batching doesn't quite seem to work yet

## Future Work
- Allow passing device pointers and CUDA streams for stream synchronization events
- Provide better examples

## Credits
Much of the C++ code is based on the [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) repo.
