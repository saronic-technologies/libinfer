#pragma once

#include "NvInfer.h"
#include <chrono>
#include <cstdlib>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <fstream>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "rust/cxx.h"
#include <unordered_map>

struct Options;
struct TensorInfo;
struct TensorInstance;
enum class TensorDataType : uint8_t;

class Logger : public nvinfer1::ILogger {
public:
  nvinfer1::ILogger::Severity reportableSeverity;

  explicit Logger() {
    const char *rustLogLevelEnvVar = "RUST_LOG";
    const char *rustLogLevel = getenv(rustLogLevelEnvVar);
    if (rustLogLevel == nullptr) {
      reportableSeverity = nvinfer1::ILogger::Severity::kWARNING;
    } else {

      std::string rustLogLevelStr(rustLogLevel);
      if (rustLogLevelStr == "error") {
        reportableSeverity = nvinfer1::ILogger::Severity::kERROR;
      } else if (rustLogLevelStr == "warn") {
        reportableSeverity = nvinfer1::ILogger::Severity::kWARNING;
      } else if (rustLogLevelStr == "info") {
        reportableSeverity = nvinfer1::ILogger::Severity::kINFO;
      } else if (rustLogLevelStr == "debug" || rustLogLevelStr == "trace") {
        reportableSeverity = nvinfer1::ILogger::Severity::kVERBOSE;
      }
    }
  }

  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override {
    if (severity > reportableSeverity) {
      return;
    }
    switch (severity) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
      spdlog::critical(msg);
      break;
    case nvinfer1::ILogger::Severity::kERROR:
      spdlog::error(msg);
      break;
    case nvinfer1::ILogger::Severity::kWARNING:
      spdlog::warn(msg);
      break;
    case nvinfer1::ILogger::Severity::kINFO:
      spdlog::info(msg);
      break;
    default:
      spdlog::debug(msg);
      break;
    }
  }
};

class Engine {
public:
  Engine(const Options &options);
  ~Engine();

  // Load and prepare the network for inference.
  void load();

  // Run inference and return output tensors (synchronous).
  rust::Vec<TensorInstance> infer(const rust::Vec<TensorInstance> &input);
  
  // Run asynchronous inference - returns immediately, use wait_for_completion() to get results
  void infer_async(const rust::Vec<TensorInstance> &input);
  
  // Wait for async inference completion and return results
  rust::Vec<TensorInstance> wait_for_completion();
  
  // Enable CUDA Graph optimization (call after first inference)
  void enable_cuda_graphs();
  
  // Enable or disable registering host memory (inputs/outputs) as pinned for faster H2D/D2H
  // When enabled, input host buffers passed by the caller are temporarily registered and
  // unregistered automatically after H2D completes. Output buffers allocated by the engine
  // are registered and unregistered after wait_for_completion().
  void enable_pinned_memory(bool enable);

  // Enable or disable additional runtime validation (e.g., inferShapes). Default is enabled.
  void set_validation_enabled(bool enable);
  
  // Check if async inference is complete (non-blocking)
  bool is_inference_complete() const;

  // Get dimensions for all input tensors
  rust::Vec<TensorInfo> get_input_tensor_info() const;

  // Get dimensions for all output tensors
  rust::Vec<TensorInfo> get_output_tensor_info() const;

private:
  // Async inference state
  std::vector<TensorInstance> mPendingOutputs;
  bool mAsyncInferenceActive;
  
  // Host memory registration control
  bool mUsePinnedHostMemory;
  bool mValidateShapes;

  // Persistent CUDA events reused across inferences
  cudaEvent_t mEventH2DComplete;
  cudaEvent_t mEventComputeComplete;
  // Output host buffers registered when pinned memory is enabled
  std::vector<void*> mRegisteredOutputHostPtrs;
  
  // Performance optimization helpers
  void setupStreams();
  void captureGraph();
  void synchronizeStreams();
  
  // Tensor metadata stored at construction time
  struct TensorMetadata {
    std::string name;
    nvinfer1::TensorIOMode ioMode;
    nvinfer1::DataType dataType;
    size_t dataTypeSize;
    nvinfer1::Dims shape;
    nvinfer1::Dims minShape;
    nvinfer1::Dims optShape;
    nvinfer1::Dims maxShape;
    size_t nonDynamicSize; // number of bytes when all dynamic dimensions are set to 1
    int32_t bufferIndex;
  };

  // Holds pointers to the input and output GPU buffers
  std::vector<void *> mBuffers;
  std::vector<TensorMetadata> mTensorMetadata; // Cached tensor metadata

  // Must keep IRuntime around for inference, see:
  // https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
  std::unique_ptr<nvinfer1::IRuntime> mRuntime = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine> mEngine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> mContext = nullptr;
  Logger mLogger;

  // Multi-stream architecture for better performance
  cudaStream_t mHostToDeviceStream;  // Stream for H2D memory transfers
  cudaStream_t mComputeStream;       // Stream for inference execution
  cudaStream_t mDeviceToHostStream;  // Stream for D2H memory transfers
  
  // CUDA Graph support for TensorRT 10.9
  cudaGraph_t mCudaGraph;
  cudaGraphExec_t mCudaGraphExec;
  bool mGraphCaptured;
  bool mUseGraphOptimization;

  // Options values.
  const std::string kEnginePath;
  const uint32_t kDeviceIndex; 
};

// Rust friends.
std::unique_ptr<Engine> load_engine(const Options &options);
