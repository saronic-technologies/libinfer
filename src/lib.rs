#[cxx::bridge]
pub mod ffi {

    #[derive(Debug, Clone)]
    /// What input data type this network accepts.
    enum InputDataType {
        UINT8,
        FP32,
    }

    #[derive(Debug, Clone)]
    /// Options for creating the inference engine.
    struct Options {
        /// Full path to the engine to load.
        path: String,

        /// Index of the device to run the engine on.
        /// If building an engine, it will be built for this device.
        /// Refer to output from e.g. `nvidia-smi` for this value.
        device_index: u32,
    }

    unsafe extern "C++" {
        include!("libinfer/src/engine.h");

        type Engine;

        /// Load the engine from the passed options.
        fn load_engine(options: &Options) -> Result<UniquePtr<Engine>>;

        /// Return the input dimensions of the engine, not including the batch dimension.
        fn get_input_dims(self: &Engine) -> Vec<u32>;

        /// Return the minimum, optimized, and maximum batch dimension for this engine.
        fn _get_batch_dims(self: &Engine) -> Vec<u32>;

        /// Return output dimensions of the engine, not including batch dimension.
        /// The meaning of these dimensions is dependent on the network definition.
        fn get_output_dims(self: &Engine) -> Vec<u32>;

        /// Return the expected length of the output feature vector.
        /// This is equivalent to multiplying the elements of `get_output_dim`.
        fn get_output_len(self: &Engine) -> u32;

        /// Returns the input data type in use by this engine.
        fn get_input_data_type(self: &Engine) -> InputDataType;

        /// Run inference on an input batch.
        ///
        /// The input batch dimension is dependent on whether the engine has been built with fixed
        /// or dynamic input batch sizes. If fixed, the input batch dimensions
        /// must match the value returned by `get_input_dim`. Dynamic may accept any input batch size.
        ///
        /// The input vector must be a flattened representation of shape
        /// `get_input_dim` with appropriate batch dimension. Likewise, the output dimension will
        /// be of shape `get_output_dim` with batch dimension equal to input batch dimension.
        fn infer(self: Pin<&mut Engine>, input: &Vec<u8>) -> Result<Vec<f32>>;
    }
}

pub use crate::ffi::Engine;
pub use crate::ffi::InputDataType;
pub use crate::ffi::Options;

use cxx::{Exception, UniquePtr};

#[derive(Debug, Clone)]
pub struct BatchDims {
    pub min: u32,
    pub opt: u32,
    pub max: u32,
}

impl Engine {
    pub fn new(options: &Options) -> Result<UniquePtr<Engine>, Exception> {
        crate::ffi::load_engine(&options)
    }

    pub fn get_batch_dims(self: &Engine) -> BatchDims {
        let vs = self._get_batch_dims();
        BatchDims {
            min: vs[0],
            opt: vs[1],
            max: vs[2],
        }
    }
}

// Engine is not thread safe, but can be moved.
unsafe impl Send for ffi::Engine {}
