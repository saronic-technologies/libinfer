#pragma once

#include "NvInfer.h"
#include <chrono>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "rust/cxx.h"

struct Options;
enum class InputDataType : uint8_t;

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

  // Run inference and return output tensor.
  rust::Vec<float> infer(const rust::Vec<TensorInput> &input);

  rust::Vec<uint32_t> get_input_dims() const {
    rust::Vec<uint32_t> rv;
    for (size_t i = 0; i < kInputDims.size(); ++i) {
      rv.push_back(mInputDims[0].d[i]);
    }
    return rv;
  };

  rust::Vec<uint32_t> _get_batch_dims() const {
    rust::Vec<uint32_t> rv;
    rv.push_back(mMinBatchSize);
    rv.push_back(mOptBatchSize);
    rv.push_back(mMaxBatchSize);
    return rv;
  }

  rust::Vec<uint32_t> get_output_dims() const {
    rust::Vec<uint32_t> rv;
    for (size_t i = 1; i < kOutputDims.size(); ++i) {
      rv.push_back(mOutputDims[0].d[i]);
    }
    return rv;
  };

  uint32_t get_output_len() const { return mOutputLengths[0]; }

  InputDataType get_input_data_type() const { return mInputDataType; }

private:
  // Holds pointers to the input and output GPU buffers
  std::vector<void *> mBuffers;
  std::vector<uint32_t> mOutputLengths{};
  std::vector<nvinfer1::Dims> mInputDims;
  std::vector<nvinfer1::Dims> mOutputDims;
  std::vector<std::string> mIOTensorNames;
  int32_t mMinBatchSize;
  int32_t mOptBatchSize;
  int32_t mMaxBatchSize;
  InputDataType mInputDataType;
  uint8_t mInputDataTypeSize;

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
  const std::vector<uint32_t> kInputDims;
  const std::vector<uint32_t> kOutputDims; 
};

// Rust friends.
std::unique_ptr<Engine> load_engine(const Options &options);
