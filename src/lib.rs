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
//! use libinfer::{Engine, Options};
//!
//! // Create engine options
//! let options = Options {
//!     path: "path/to/model.engine".into(),
//!     device_index: 0,
//!     input_shapes: vec![3, 224, 224], // Example input shape for an image model without batch dimension
//!     output_shapes: vec![300, 13], // Example output shape for a classification model without batch dimension
//! };
//!
//! // Load the engine
//! let mut engine = Engine::new(&options).unwrap();
//!
//! // Get input dimensions and prepare input data
//! let dims = engine.get_input_dims();
//! let input_size = dims.iter().fold(1, |acc, &e| acc * e as usize);
//! let input = vec![0u8; input_size];
//!
//! // Run inference
//! let output = engine.pin_mut().infer(&input).unwrap();
//!
//! // Process output...
//! ```

#[cxx::bridge]
pub mod ffi {

    #[derive(Debug, Clone)]
    /// What input data type this network accepts.
    enum InputDataType {
        /// 8-bit unsigned integer input type
        UINT8,
        /// 32-bit floating point input type
        FP32,
    }

    #[derive(Debug, Clone)]
    /// Input shape info
    struct ShapeInfo {
        /// Tensor input/output name
        name: String,
        /// Tensor input/output shape
        dims: Vec<u32>,
    }

    #[derive(Debug, Clone)]
    /// Tensor input class
    struct TensorInput {
        name: String,
        tensor: Vec<u8>,
    }
    
    #[derive(Debug, Clone)]
    /// Tensor output class
    struct TensorOutput {
        name: String,
        data: Vec<f32>,
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

        /// Input shapes of the model, excluding batch dimensions.
        /// This should match the input tensor shape expected by the model.
        /// For example, for an image model, this might be [[3, 224, 224]]
        input_shape: Vec<ShapeInfo>,

        /// Output shape of the model, excluding batch dimension.
        /// This should match the output tensor shape produced by the model.
        /// For example, for rtdetr, this might be [[300, 13]]
        output_shape: Vec<ShapeInfo>,
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

        /// Return the input dimensions of the engine, not including the batch dimension.
        ///
        /// # Returns
        /// A vector of dimensions in the format [Channels, Height, Width] for image inputs,
        /// or the appropriate dimensions for the model's input tensor.
        fn get_input_dims(self: &Engine) -> Vec<u32>;

        /// Return the minimum, optimized, and maximum batch dimension for this engine.
        /// This is an internal function used by `get_batch_dims`.
        fn _get_batch_dims(self: &Engine) -> Vec<u32>;

        /// Return output dimensions of the engine, not including batch dimension.
        ///
        /// # Returns
        /// A vector representing the output tensor dimensions. The meaning of these
        /// dimensions is dependent on the network definition.
        fn get_output_dims(self: &Engine) -> Vec<u32>;

        /// Return the expected length of the output feature vector.
        ///
        /// # Returns
        /// The total number of elements in the output tensor, equivalent to
        /// multiplying all elements of `get_output_dims`.
        fn get_output_len(self: &Engine) -> u32;

        /// Returns the input data type expected by this engine.
        ///
        /// # Returns
        /// The input data type (UINT8 or FP32) that this model expects.
        fn get_input_data_type(self: &Engine) -> InputDataType;

        /// Run inference on an input batch.
        ///
        /// # Arguments
        /// * `input` - A flattened vector representing the input tensor data
        ///
        /// # Returns
        /// A Result containing either the output tensor data as a vector of f32 values,
        /// or an error message if inference failed.
        ///
        /// # Details
        /// The input batch dimension is dependent on whether the engine has been built with fixed
        /// or dynamic input batch sizes. If fixed, the input batch dimensions
        /// must match the value returned by `get_input_dims`. Dynamic may accept any input batch size
        /// within the range specified by `get_batch_dims`.
        ///
        /// The input vector must be a flattened representation of shape
        /// `get_input_dims` with appropriate batch dimension. Likewise, the output dimension will
        /// be of shape `get_output_dims` with batch dimension equal to input batch dimension.
        fn infer(self: Pin<&mut Engine>, input: &Vec<TensorInput>) -> Result<Vec<TensorOutput>>;
    }
}

pub use crate::ffi::Engine;
pub use crate::ffi::InputDataType;
pub use crate::ffi::Options;

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

    /// Get the batch dimension constraints for this engine.
    ///
    /// # Returns
    /// A `BatchDims` struct containing the minimum, optimal, and maximum
    /// batch sizes supported by this engine.
    ///
    /// For fixed-batch engines, all values will typically be the same.
    /// For dynamic-batch engines, these values represent the valid range
    /// of batch sizes that can be used.
    pub fn get_batch_dims(self: &Engine) -> BatchDims {
        let vs = self._get_batch_dims();
        BatchDims {
            min: vs[0],
            opt: vs[1],
            max: vs[2],
        }
    }
}

// Engine is not thread safe, but can be moved between threads.
unsafe impl Send for ffi::Engine {}
