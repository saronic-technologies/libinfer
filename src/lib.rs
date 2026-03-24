//! # libinfer
//!
//! Rust interface to TensorRT engines. Caller provides device memory
//! and CUDA streams.

use cudarc::driver::sys::{CUdeviceptr, CUstream};
use cxx::{Exception, UniquePtr};

#[cxx::bridge]
mod ffi {
    #[derive(Debug, Clone)]
    enum TensorDataType {
        UINT8,
        FP32,
        INT64,
        BOOL,
    }

    #[derive(Debug, Clone)]
    struct TensorInfo {
        name: String,
        dims: Vec<u32>,
        dtype: TensorDataType,
    }

    #[derive(Debug, Clone)]
    struct Options {
        path: String,
    }

    unsafe extern "C++" {
        include!("libinfer/src/engine.h");

        type Engine;

        fn load_engine(options: &Options) -> Result<UniquePtr<Engine>>;

        fn get_input_dims(self: &Engine) -> Vec<TensorInfo>;
        fn get_output_dims(self: &Engine) -> Vec<TensorInfo>;
        fn _get_batch_dims(self: &Engine) -> Vec<u32>;
        fn get_output_len(self: &Engine) -> u32;
        fn get_num_inputs(self: &Engine) -> usize;
        fn get_num_outputs(self: &Engine) -> usize;

        /// # Safety
        ///
        /// Pointer arrays and stream must be valid. Device memory must be
        /// large enough for the given batch_size.
        unsafe fn infer(
            self: Pin<&mut Engine>,
            input_ptrs: *const u64,
            num_inputs: usize,
            output_ptrs: *const u64,
            num_outputs: usize,
            stream: u64,
            batch_size: u32,
        ) -> Result<()>;

        /// # Safety
        ///
        /// Same as `infer`. Caller must synchronize the stream before
        /// reading outputs.
        unsafe fn infer_async(
            self: Pin<&mut Engine>,
            input_ptrs: *const u64,
            num_inputs: usize,
            output_ptrs: *const u64,
            num_outputs: usize,
            stream: u64,
            batch_size: u32,
        ) -> Result<()>;
    }
}

pub use ffi::{Options, TensorDataType, TensorInfo};

impl TensorDataType {
    /// Size in bytes of a single element of this type.
    pub fn byte_size(&self) -> usize {
        if *self == TensorDataType::FP32 {
            4
        } else if *self == TensorDataType::INT64 {
            8
        } else {
            1
        }
    }
}

impl TensorInfo {
    /// Total number of elements per sample (excludes batch dimension).
    pub fn elem_count(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Total byte size per sample.
    pub fn byte_size(&self) -> usize {
        self.elem_count() * self.dtype.byte_size()
    }
}

unsafe impl Send for ffi::Engine {}

/// Batch dimension constraints.
#[derive(Debug, Clone)]
pub struct BatchDims {
    /// Minimum supported batch size.
    pub min: u32,
    /// Optimal batch size.
    pub opt: u32,
    /// Maximum supported batch size.
    pub max: u32,
}

/// TensorRT inference engine.
pub struct Engine {
    inner: UniquePtr<ffi::Engine>,
}

impl Engine {
    /// Load a TensorRT engine. The caller must ensure the correct CUDA
    /// device is current before calling this, and that the CUDA context
    /// outlives the engine.
    pub fn new(options: &Options) -> Result<Self, Exception> {
        let inner = ffi::load_engine(options)?;
        Ok(Self { inner })
    }

    /// Run inference synchronously. Blocks until complete.
    pub fn infer(
        &mut self,
        inputs: &[CUdeviceptr],
        outputs: &[CUdeviceptr],
        stream: CUstream,
        batch_size: u32,
    ) -> Result<(), Exception> {
        unsafe {
            self.inner.pin_mut().infer(
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_ptr(),
                outputs.len(),
                stream as u64,
                batch_size,
            )
        }
    }

    /// Enqueue inference and return immediately.
    pub fn infer_async(
        &mut self,
        inputs: &[CUdeviceptr],
        outputs: &[CUdeviceptr],
        stream: CUstream,
        batch_size: u32,
    ) -> Result<(), Exception> {
        unsafe {
            self.inner.pin_mut().infer_async(
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_ptr(),
                outputs.len(),
                stream as u64,
                batch_size,
            )
        }
    }

    /// Get input tensor metadata.
    pub fn get_input_dims(&self) -> Vec<TensorInfo> {
        self.inner.get_input_dims().into_iter().collect()
    }

    /// Get output tensor metadata.
    pub fn get_output_dims(&self) -> Vec<TensorInfo> {
        self.inner.get_output_dims().into_iter().collect()
    }

    /// Get batch dimension constraints.
    pub fn get_batch_dims(&self) -> BatchDims {
        let vs = self.inner._get_batch_dims();
        BatchDims {
            min: vs[0],
            opt: vs[1],
            max: vs[2],
        }
    }

    /// Number of input tensors.
    pub fn get_num_inputs(&self) -> usize {
        self.inner.get_num_inputs()
    }

    /// Number of output tensors.
    pub fn get_num_outputs(&self) -> usize {
        self.inner.get_num_outputs()
    }

    /// Output length (elements) of the first output tensor.
    pub fn get_output_len(&self) -> u32 {
        self.inner.get_output_len()
    }
}

unsafe impl Send for Engine {}
