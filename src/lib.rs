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
        /// Only 1 is currently allowed.
        optimized_batch_size: i32,

        /// The maximum allowable batch size.
        /// Only 1 is currently allowed.
        max_batch_size: i32,
    }

    unsafe extern "C++" {
        include!("libinfer/src/engine.h");

        type Engine;

        /// Build the engine from the passed options.
        fn make_engine(options: &Options) -> Result<UniquePtr<Engine>>;

        /// Return the input dimensions of the engine.
        /// Values are in CHW order.
        /// Note: batch dimension is implicitly always 1 in the current implementation.
        fn get_input_dim(engine: &UniquePtr<Engine>) -> Vec<u32>;

        /// Return output dimensions of the network.
        /// The meaning of these dimensions is dependant on the network definition.
        fn get_output_dim(engine: &UniquePtr<Engine>) -> Vec<u32>;

        /// Return the expected length of the output feature vector.
        /// This is equivalent to multiplying the outputs of `get_output_dim`.
        fn get_output_len(engine: &UniquePtr<Engine>) -> u32;

        /// Run inference on an input vector.
        /// This vector should be a flattened representation of `get_input_dim`, in CHW order.
        fn run_inference(engine: &UniquePtr<Engine>, input: &Vec<f32>) -> Result<Vec<f32>>;
    }
}

// Engine is not thread safe, but can be moved.
unsafe impl Send for ffi::Engine {}
