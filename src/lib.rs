//! # libinfer
//!
//! A Rust interface to TensorRT engines for high-performance ML inference on NVIDIA GPUs.
//!
//! This crate provides a safe, idiomatic Rust API for TensorRT, allowing easy integration
//! of GPU-accelerated machine learning models into Rust applications.
//!
//! ## Example
//!
//! ```rust,no_run
//! use libinfer::{Engine, Options, InputTensor, TensorOutput, TensorInfo};
//!
//! // Create engine options
//! let options = Options {
//!     path: "path/to/model.engine".into(),
//!     device_index: 0,
//! };
//!
//! // Load the engine
//! let mut engine = Engine::new(&options).unwrap();
//!
//! // Get input dimensions for all tensors
//! let input_dims: Vec<TensorInfo> = engine.get_input_dims();
//! 
//! // Create input tensors
//! let mut input_tensors: Vec<InputTensor> = Vec::new();
//! for tensor_info in input_dims {
//!     let size = tensor_info.dims.iter().fold(1, |acc, &d| acc * d as usize);
//!     let input_tensor = InputTensor {
//!         name: tensor_info.name,
//!         tensor: vec![0u8; size],
//!     };
//!     input_tensors.push(input_tensor);
//! }
//!
//! // Run inference
//! let outputs: Vec<TensorOutput> = engine.pin_mut().infer(&input_tensors).unwrap();
//!
//! // Process outputs...
//! for output in outputs {
//!     println!("Output '{}': {} elements", output.name, output.data.len());
//! }
//! ```

#[cxx::bridge]
pub mod ffi {

    #[derive(Debug, Clone)]
    /// What input data type this network accepts.
    enum TensorDataType {
        /// 8-bit unsigned integer input type
        UINT8,
        /// 32-bit floating point input type
        FP32,
        /// 64-bit integer input type
        INT64,
        /// 8-bit boolean input type
        BOOL,
    }


    #[derive(Debug, Clone)]
    /// Tensor dimension information
    struct TensorInfo {
        name: String,
        dims: Vec<u32>,
        dtype: TensorDataType,
    }

    #[derive(Debug, Clone)]
    /// Tensor input class
    struct InputTensor {
        name: String,
        data: Vec<u8>,
        dtype: TensorDataType,
    }
    
    #[derive(Debug, Clone)]
    /// Tensor output class
    struct OutputTensor {
        name: String,
        data: Vec<u8>,
        dtype: TensorDataType,
    }

    #[derive(Debug, Clone)]
    /// Per-input shape profile from TensorRT optimization profile.
    struct InputShapeProfile {
        name: String,
        has_dynamic_shape: bool,
        min_shape: Vec<i32>,
        opt_shape: Vec<i32>,
        max_shape: Vec<i32>,
    }

    #[derive(Debug, Clone)]
    /// Options for creating the inference engine.
    struct Options {
        /// Full path to the TensorRT engine file to load.
        /// This should be a pre-built engine file (.engine) for the target device.
        path: String,

        /// Index of the NVIDIA GPU device to run the engine on.
        /// If building an engine, it will be built for this device.
        /// Refer to output from e.g. `nvidia-smi` for this value.
        /// A value of 0 typically refers to the first GPU.
        device_index: u32,
    }

    extern "Rust" {
        /// Allocate a zero-initialized Vec<u8> of `size` bytes on the Rust heap.
        fn new_output_buffer(size: usize) -> Vec<u8>;
    }

    unsafe extern "C++" {
        include!("libinfer/src/engine.h");

        /// TensorRT engine wrapper for executing inference
        type Engine;

        /// Load the TensorRT engine from the passed options.
        ///
        /// # Returns
        /// A Result containing either the loaded engine or an error message.
        fn load_engine(options: &Options) -> Result<UniquePtr<Engine>>;

        /// Return the input dimensions of all input tensors, not including the batch dimension.
        ///
        /// # Returns
        /// A vector of TensorInfo containing name and dimensions for each input tensor.
        fn get_input_dims(self: &Engine) -> Vec<TensorInfo>;

        /// Return per-input shape profiles (min/opt/max) from the TensorRT optimization profile.
        /// This is an internal function used by `get_input_shape_profiles`.
        fn _get_input_shape_profiles(self: &Engine) -> Vec<InputShapeProfile>;

        /// Return output dimensions of all output tensors, not including batch dimension.
        ///
        /// # Returns
        /// A vector of TensorInfo containing name and dimensions for each output tensor.
        fn get_output_dims(self: &Engine) -> Vec<TensorInfo>;

        /// Get the number of input tensors.
        fn get_num_inputs(self: &Engine) -> usize;

        /// Get the number of output tensors.
        fn get_num_outputs(self: &Engine) -> usize;

        /// Run inference on the provided input tensors.
        ///
        /// # Arguments
        /// * `input` - Named input tensors with flattened data bytes
        ///
        /// # Returns
        /// A Result containing the output tensors or an error message.
        ///
        /// # Details
        /// Each input tensor's batch size is resolved independently from its data size.
        /// For dynamic inputs, the batch dimension must fall within the range specified
        /// by `get_input_shape_profiles`. For static inputs, the data size must match
        /// the fixed shape exactly. Different inputs may have different batch sizes.
        fn infer(self: Pin<&mut Engine>, input: &Vec<InputTensor>) -> Result<Vec<OutputTensor>>;
    }
}

// Primary exports
pub use crate::ffi::{
    Engine,
    TensorDataType,
    Options,
    TensorInfo,
    InputTensor,
    OutputTensor,
    InputShapeProfile,
};

use cxx::{Exception, UniquePtr};

/// Represents the batch dimensions supported by a TensorRT engine.
///
/// Deprecated: Use `get_input_shape_profiles()` for per-input profiles.
#[derive(Debug, Clone)]
pub struct BatchDims {
    /// Minimum supported batch size
    pub min: u32,
    /// Optimal (default) batch size for best performance
    pub opt: u32,
    /// Maximum supported batch size
    pub max: u32,
}

/// Per-input shape profile from a TensorRT optimization profile.
#[derive(Debug, Clone)]
pub struct ShapeProfile {
    /// Name of the input tensor
    pub name: String,
    /// Whether this input has any dynamic dimensions
    pub has_dynamic_shape: bool,
    /// Minimum shape (full dims including batch) from the optimization profile
    pub min_shape: Vec<i32>,
    /// Optimal shape (full dims including batch) from the optimization profile
    pub opt_shape: Vec<i32>,
    /// Maximum shape (full dims including batch) from the optimization profile
    pub max_shape: Vec<i32>,
}

impl Engine {
    /// Create a new TensorRT engine from the given options.
    ///
    /// # Arguments
    /// * `options` - Configuration options for the engine
    ///
    /// # Returns
    /// A Result containing either the initialized engine or an error
    pub fn new(options: &Options) -> Result<UniquePtr<Engine>, Exception> {
        crate::ffi::load_engine(&options)
    }

    /// Get shape profiles for all input tensors.
    ///
    /// Each input tensor has its own min/opt/max shape from the TensorRT
    /// optimization profile. For static inputs, min == opt == max.
    /// For dynamic inputs, the shapes represent the valid range.
    pub fn get_input_shape_profiles(&self) -> Vec<ShapeProfile> {
        self._get_input_shape_profiles()
            .into_iter()
            .map(|p| ShapeProfile {
                name: p.name,
                has_dynamic_shape: p.has_dynamic_shape,
                min_shape: p.min_shape,
                opt_shape: p.opt_shape,
                max_shape: p.max_shape,
            })
            .collect()
    }

    /// Get the batch dimension constraints for this engine.
    ///
    /// Deprecated: Use `get_input_shape_profiles()` for per-input profiles.
    /// This returns the profile of the first dynamic input's first dimension,
    /// or (1, 1, 1) if no inputs are dynamic.
    pub fn get_batch_dims(&self) -> BatchDims {
        let profiles = self._get_input_shape_profiles();
        for p in &profiles {
            if p.has_dynamic_shape {
                return BatchDims {
                    min: p.min_shape[0] as u32,
                    opt: p.opt_shape[0] as u32,
                    max: p.max_shape[0] as u32,
                };
            }
        }
        BatchDims { min: 1, opt: 1, max: 1 }
    }
}

fn new_output_buffer(size: usize) -> Vec<u8> {
    vec![0u8; size]
}

// Engine is not thread safe, but can be moved between threads.
unsafe impl Send for ffi::Engine {}
