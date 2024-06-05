#pragma once

#include "NvInfer.h"
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "rust/cxx.h"

struct Options;

class Logger : public nvinfer1::ILogger {
public:
  nvinfer1::ILogger::Severity reportableSeverity;

  explicit Logger(
      nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
      : reportableSeverity(severity) {}

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

  // Build the network.
  void build();

  // Load and prepare the network for inference.
  void load();

  // Run inference and return output tensor.
  rust::Vec<float> infer(const rust::Vec<float> &input);

  rust::Vec<uint32_t> getInputDims() const {
    rust::Vec<uint32_t> rv;
    rv.push_back(mInputBatchSize);
    for (int i = 0; i < 3; ++i) {
      rv.push_back(mInputDims[0].d[i]);
    }
    return rv;
  };

  rust::Vec<uint32_t> getOutputDims() const {
    rust::Vec<uint32_t> rv;
    for (int i = 0; i < 3; ++i) {
      rv.push_back(mOutputDims[0].d[i]);
    }
    return rv;
  };

  uint32_t getOutputLen() const { return mOutputLengths[0]; }

private:
  // Converts the engine options into a string.
  std::string serializeEngineOptions(const Options &options);

  void getDeviceNames(std::vector<std::string> &deviceNames);

  // Holds pointers to the input and output GPU buffers
  std::vector<void *> mBuffers;
  std::vector<uint32_t> mOutputLengths{};
  std::vector<nvinfer1::Dims3> mInputDims;
  std::vector<nvinfer1::Dims> mOutputDims;
  std::vector<std::string> mIOTensorNames;
  int32_t mInputBatchSize;

  // Must keep IRuntime around for inference, see:
  // https://forums.developer.nvidia.com/t/is-it-safe-to-deallocate-nvinfer1-iruntime-after-creating-an-nvinfer1-icudaengine-but-before-running-inference-with-said-icudaengine/255381/2?u=cyruspk4w6
  std::unique_ptr<nvinfer1::IRuntime> mRuntime = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine> mEngine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> mContext = nullptr;
  Logger mLogger;

  std::string mEnginePath;

  // Option values.
  const std::string kModelName;
  const std::string kSearchPath;
  const std::string kSavePath;
  const std::string kCanonicalEngineName;
  const uint8_t kPrecision;
  const uint32_t kDeviceIndex;
  const int32_t kOptBatchSize;
  const int32_t kMaxBatchSize;
};

// Rust friends.
std::unique_ptr<Engine> make_engine(const Options &options);

rust::Vec<float> run_inference(const std::unique_ptr<Engine> &engine,
                               const rust::Vec<float> &input);

rust::Vec<uint32_t> get_input_dim(const std::unique_ptr<Engine> &engine);
rust::Vec<uint32_t> get_output_dim(const std::unique_ptr<Engine> &engine);
uint32_t get_output_len(const std::unique_ptr<Engine> &engine);
