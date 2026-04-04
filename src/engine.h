#pragma once

#include "NvInfer.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

#include "rust/cxx.h"

struct Options;
struct TensorInfo;
enum class TensorDataType : uint8_t;

class Logger : public nvinfer1::ILogger {
public:
  nvinfer1::ILogger::Severity reportableSeverity;

  explicit Logger() {
    const char *rustLogLevel = getenv("RUST_LOG");
    if (rustLogLevel == nullptr) {
      reportableSeverity = nvinfer1::ILogger::Severity::kWARNING;
    } else {
      std::string level(rustLogLevel);
      if (level == "error") {
        reportableSeverity = nvinfer1::ILogger::Severity::kERROR;
      } else if (level == "warn") {
        reportableSeverity = nvinfer1::ILogger::Severity::kWARNING;
      } else if (level == "info") {
        reportableSeverity = nvinfer1::ILogger::Severity::kINFO;
      } else if (level == "debug" || level == "trace") {
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
  ~Engine() = default;

  void load();

  // Enqueue inference with caller-provided device pointers and stream.
  // Synchronizes the stream before returning.
  void infer(const uint64_t *input_ptrs, size_t num_inputs,
             const uint64_t *output_ptrs, size_t num_outputs,
             uint64_t stream, uint32_t batch_size);

  // Same as infer() but does not synchronize. Caller is responsible.
  void infer_async(const uint64_t *input_ptrs, size_t num_inputs,
                   const uint64_t *output_ptrs, size_t num_outputs,
                   uint64_t stream, uint32_t batch_size);

  rust::Vec<TensorInfo> get_input_dims() const;
  rust::Vec<TensorInfo> get_output_dims() const;
  rust::Vec<uint32_t> _get_batch_dims() const;
  uint32_t get_output_len() const;
  size_t get_num_inputs() const;
  size_t get_num_outputs() const;

private:
  struct TensorMetadata {
    std::string name;
    nvinfer1::TensorIOMode ioMode;
    nvinfer1::DataType dataType;
    size_t dataTypeSize;
    nvinfer1::Dims dims;
  };

  void enqueue(const uint64_t *input_ptrs, size_t num_inputs,
               const uint64_t *output_ptrs, size_t num_outputs,
               cudaStream_t stream, uint32_t batch_size);

  std::vector<TensorMetadata> mTensorMetadata;
  std::vector<uint32_t> mOutputLengths;
  int32_t mMinBatchSize = 0;
  int32_t mOptBatchSize = 0;
  int32_t mMaxBatchSize = 0;

  std::unique_ptr<nvinfer1::IRuntime> mRuntime;
  std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
  std::unique_ptr<nvinfer1::IExecutionContext> mContext;
  Logger mLogger;

  const std::string kEnginePath;
};

std::unique_ptr<Engine> load_engine(const Options &options);
