# `libinfer`
This library provides a simple Rust interface to a TensorRT engine using [cxx]([url](https://cxx.rs/))

## Usage
The goal of the API is to keep as much processing in Rust land as possible. Here is a sample usage.

```
// Look for an engine file appropriate for our device.
// If not found, try looking for "test/yolov8n.onnx" and build a new engine.
let options = Options {
    model_name: "yolov8n".into(),
    search_path: "test".into(),
    save_path: "test".into(),
    device_index: 0,
    precision: Precision::FP16,
    optimized_batch_size: 1,
    max_batch_size: 1
};
let engine = make_engine(&options).unwrap();

// Get input dimensions of the engine as [Channels, Height, Width].
// For now, only a batch size of 1 is supported.
let dims = get_input_dim(&engine);

// Construct our input.
// This is fed directly into the network, so it must match the expected format.
// In TensorRT, this is [B, C, H, W] of `f32`s.
let input = vec![0.0; dims[0] * dims[1] * dims[2]];

// Run inference.
let output = run_inference(&engine).unwrap();

// Postprocess the output according to your model's output format.
...
```

## Current Limitations
- Only batch size of 1 is currently supported
- Only fp16 quantization is currently supported
- The underlying engine code is not threadsafe (and the Rust binding does not implement `Sync`)
- Additional benchmarking and testing is desirable

## Credits
Much of the C++ code is based on the [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) repo.