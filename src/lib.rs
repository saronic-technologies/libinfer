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
//! use libinfer::{Engine, Options, TensorInstance, TensorInfo};
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
//! let input_dims: Vec<TensorInfo> = engine.get_input_tensor_info();
//! 
//! // Create input tensors
//! let mut input_tensors: Vec<TensorInstance> = Vec::new();
//! for tensor_info in input_dims {
//!     let size = tensor_info.dims.iter().fold(1, |acc, &d| acc * d as usize);
//!     let input_tensor = TensorInstance {
//!         name: tensor_info.name,
//!         tensor: vec![0u8; size],
//!     };
//!     input_tensors.push(input_tensor);
//! }
//!
//! // Run inference
//! let outputs: Vec<TensorInstance> = engine.pin_mut().infer(&input_tensors).unwrap();
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
        shape: Vec<i64>, // -1 for dynamic dimensions
        dtype: TensorDataType,
        min_shape: Vec<i64>, // min shape for dynamic dims
        opt_shape: Vec<i64>, // opt shape for dynamic dims
        max_shape: Vec<i64>, // max shape for dynamic dims
    }

    #[derive(Debug, Clone)]
    /// Tensor input class
    struct TensorInstance {
        name: String,
        data: Vec<u8>,
        shape: Vec<i64>, // this should always be positive, just i64 for convenience
        dtype: TensorDataType,
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
        fn get_input_tensor_info(self: &Engine) -> Vec<TensorInfo>;

        /// Return output dimensions of all output tensors, not including batch dimension.
        ///
        /// # Returns
        /// A vector of TensorInfo containing name and dimensions for each output tensor.
        fn get_output_tensor_info(self: &Engine) -> Vec<TensorInfo>;

        /// Run inference on an input batch.
        ///
        /// # Arguments
        /// * `input` - A flattened vector representing the input tensor data
        ///
        /// # Returns
        /// A Result containing either the output tensor data as a vector of f32 values,
        /// or an error message if inference failed.
        ///
        /// The input vector must be a flattened representation of shape
        /// `get_input_tensor_info` with appropriate dynamic dimensions. Likewise, the output dimension will
        /// be of shape `get_output_tensor_info`.
        fn infer(self: Pin<&mut Engine>, input: &Vec<TensorInstance>) -> Result<Vec<TensorInstance>>;
    }
}

// Primary exports
pub use crate::ffi::{
    Engine,
    TensorDataType,
    Options,
    TensorInfo,
    TensorInstance,
};

use cxx::{Exception, UniquePtr};

/// Represents the batch dimensions supported by a TensorRT engine
#[derive(Debug, Clone)]
pub struct BatchDims {
    /// Minimum supported batch size
    pub min: u32,
    /// Optimal (default) batch size for best performance
    pub opt: u32,
    /// Maximum supported batch size
    pub max: u32,
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
}

// Engine is not thread safe, but can be moved between threads.
unsafe impl Send for ffi::Engine {}
