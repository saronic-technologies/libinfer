#[cxx::bridge]
pub mod ffi {

    #[derive(Debug, Clone)]
    #[repr(u8)]
    /// What precision to build a new engine with.
    /// Note that `INT8` is not yet supported.
    enum Precision {
        FP32,
        FP16,
        INT8,
    }

    #[derive(Debug, Clone)]
    /// Options for creating the inference engine.
    struct Options {
        /// Name of the model.
        /// This should be the filename of either a `.engine` file or a `.onnx` file
        /// without the extension and without path specifiers.
        model_name: String,

        /// Path to search for an appropriate engine or onnx file.
        /// First an appropriate engine file will be searched for according to the
        /// following naming scheme:
        ///
        /// ```
        /// <model_name>_<device_name>_<precision>.engine
        /// ```
        ///
        /// Where `device_name` is the `name` property of the CUDA device
        /// specified by `device_index`.
        ///
        /// The engine file must match this name exactly to help ensure that only
        /// engines which have been built for the specific device are used.
        ///
        /// If an appropriate engine file is not found here or in `save_path`
        /// a file of the name `<model_name>.onnx` will attempt to be found and a new
        /// engine file built according to the specified options.
        search_path: String,

        /// Path to store an engine file. Secondary search path for loading an engine.
        /// If empty the engine will be placed in current working directory.
        save_path: String,

        /// Index of the device to run the engine on.
        /// If building an engine, it will be built for this device.
        ///
        /// Refer to output from e.g. `nvidia-smi` for this value.
        device_index: u32,

        /// Precision to build the engine with.
        precision: Precision,

        /// The batch size which should be optimized for.
        optimized_batch_size: i32,

        /// The maximum allowable batch size.
        /// If the provided onnx network has a fixed batch size,
        /// then `optimized_batch_size` and `max_batch_size` must be equal.
        max_batch_size: i32,
    }

    unsafe extern "C++" {
        include!("libinfer/src/engine.h");

        type Engine;

        /// Build the engine from the passed options.
        fn make_engine(options: &Options) -> Result<UniquePtr<Engine>>;

        /// Return the input dimensions of the engine.
        /// For image inputs, values are in BCHW order.
        /// A value of -1 in the batch dimension indicates the engine may accept a
        /// dynamic batch size.
        fn get_input_dims(self: &Engine) -> Vec<u32>;

        /// Return output dimensions of the network.
        /// The meaning of these dimensions is dependent on the network definition,
        /// but the batch dimension always comes first.
        /// A value of -1 in the batch dimension indicates the engine has dynamic batch size
        /// enabled and will the output will have a batch dimension equal to the input value.
        fn get_output_dims(self: &Engine) -> Vec<u32>;

        /// Return the expected length of the output feature vector.
        /// This is equivalent to multiplying the elements of `get_output_dim`.
        fn get_output_len(self: &Engine) -> u32;

        /// Run inference on an input batch.
        ///
        /// The input batch dimension is dependent on whether the engine has been built with fixed
        /// or dynamic input batch sizes. If fixed, the input batch dimensions
        /// must match the value returned by `get_input_dim`. Dynamic may accept any input batch size.
        ///
        /// The input vector must be a flattened representation of shape
        /// `get_input_dim` with appropriate batch dimension. Likewise, the output dimension will
        /// be of shape `get_output_dim` with batch dimension equal to input batch dimension.
        fn infer(self: Pin<&mut Engine>, input: &Vec<f32>) -> Result<Vec<f32>>;
    }
}

pub use crate::ffi::Engine;
pub use crate::ffi::Options;
pub use crate::ffi::Precision;

use cxx::{Exception, UniquePtr};

impl Engine {
    pub fn new(options: &Options) -> Result<UniquePtr<Engine>, Exception> {
        crate::ffi::make_engine(&options)
    }
}

// Engine is not thread safe, but can be moved.
unsafe impl Send for ffi::Engine {}
