#pragma once

#include "NvInfer.h"
#include <chrono>
#include <cstdlib>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <fstream>
#include <unistd.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

// RAII helper to redirect stdout to stderr.
// TensorRT sometimes prints warnings directly to stdout, bypassing the ILogger interface.
// This class ensures all stdout output goes to stderr instead.
class StdoutToStderr {
public:
    StdoutToStderr() {
        // Save original stdout
        saved_stdout = dup(STDOUT_FILENO);
        // Redirect stdout to stderr
        dup2(STDERR_FILENO, STDOUT_FILENO);
    }

    ~StdoutToStderr() {
        // Flush before restoring
        fflush(stdout);
        // Restore original stdout
        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);
    }

    // Non-copyable, non-movable
    StdoutToStderr(const StdoutToStderr&) = delete;
    StdoutToStderr& operator=(const StdoutToStderr&) = delete;

private:
    int saved_stdout;
};

#include "rust/cxx.h"

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

  // Run inference and return output tensors.
  rust::Vec<TensorInstance> infer(const rust::Vec<TensorInstance> &input);

  // Get dimensions for all input tensors
  rust::Vec<TensorInfo> get_input_tensor_info() const;

  // Get dimensions for all output tensors
  rust::Vec<TensorInfo> get_output_tensor_info() const;

private:
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

  cudaStream_t mInferenceCudaStream;

  // Options values.
  const std::string kEnginePath;
  const uint32_t kDeviceIndex; 
};

// Rust friends.
std::unique_ptr<Engine> load_engine(const Options &options);
