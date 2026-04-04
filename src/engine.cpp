#include "engine.h"

#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <NvInferRuntimeBase.h>
#include <fstream>
#include <mutex>

#include "libinfer/src/lib.rs.h"

using namespace nvinfer1;

static size_t getDataTypeSize(DataType dt) {
  switch (dt) {
  case DataType::kFLOAT:
    return 4;
  case DataType::kUINT8:
    return 1;
  case DataType::kINT64:
    return 8;
  case DataType::kBOOL:
    return 1;
  default:
    throw std::runtime_error("Unsupported tensor data type");
  }
}

static TensorDataType toTensorDataType(DataType dt) {
  switch (dt) {
  case DataType::kFLOAT:
    return TensorDataType::FP32;
  case DataType::kUINT8:
    return TensorDataType::UINT8;
  case DataType::kINT64:
    return TensorDataType::INT64;
  case DataType::kBOOL:
    return TensorDataType::BOOL;
  default:
    throw std::runtime_error("Unsupported tensor data type");
  }
}

static void checkCudaErrorCode(cudaError_t code) {
  if (code != 0) {
    std::string errMsg =
        "CUDA operation failed with code: " + std::to_string(code) + "(" +
        cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
    throw std::runtime_error(errMsg);
  }
}

template <typename T> static void resize(rust::Vec<T> &v, size_t len) {
  v.reserve(len);
  while (v.size() < len) {
    v.push_back(T{});
  }
}

std::unique_ptr<Engine> load_engine(const Options &options) {
  auto engine = std::make_unique<Engine>(options);
  engine->load();
  return engine;
}

static void init_logger() {
  spdlog::set_pattern("%+", spdlog::pattern_time_type::utc);
  spdlog::set_default_logger(spdlog::stderr_color_mt("libinfer"));

  const char *rustLogLevel = getenv("RUST_LOG");
  if (rustLogLevel == nullptr) {
    spdlog::set_level(spdlog::level::warn);
  } else {
    std::string level(rustLogLevel);
    if (level == "error") {
      spdlog::set_level(spdlog::level::err);
    } else if (level == "warn") {
      spdlog::set_level(spdlog::level::warn);
    } else if (level == "info") {
      spdlog::set_level(spdlog::level::info);
    } else if (level == "debug" || level == "trace") {
      spdlog::set_level(spdlog::level::debug);
    }
  }
}

Engine::Engine(const Options &options)
    : kEnginePath(options.path) {
  static std::once_flag logger_init;
  std::call_once(logger_init, init_logger);
}

void Engine::load() {
  initLibNvInferPlugins(&mLogger, "");

  std::ifstream file(kEnginePath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open engine file");
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (file.fail()) {
    throw std::runtime_error("Failed to seek in engine file");
  }

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Unable to read engine file");
  }

  mRuntime = std::unique_ptr<IRuntime>{createInferRuntime(mLogger)};
  if (!mRuntime) {
    throw std::runtime_error("Runtime not initialized");
  }

  // Caller is responsible for setting the CUDA device via CudaContext
  // before constructing the engine.
  mEngine = std::unique_ptr<ICudaEngine>(
      mRuntime->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!mEngine) {
    throw std::runtime_error("Engine deserialization failed");
  }

  mContext = std::unique_ptr<IExecutionContext>(
      mEngine->createExecutionContext());
  if (!mContext) {
    throw std::runtime_error("Could not create execution context");
  }

  mOutputLengths.clear();
  mTensorMetadata.clear();
  mTensorMetadata.reserve(mEngine->getNbIOTensors());

  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorShape = mEngine->getTensorShape(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);

    TensorMetadata metadata;
    metadata.name = std::string(tensorName);
    metadata.ioMode = tensorType;
    metadata.dataType = tensorDataType;
    metadata.dataTypeSize = getDataTypeSize(tensorDataType);
    metadata.dims = tensorShape;
    mTensorMetadata.push_back(std::move(metadata));

    if (tensorType == TensorIOMode::kINPUT) {
      int32_t minBatch =
          mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMIN).d[0];
      int32_t optBatch =
          mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kOPT).d[0];
      int32_t maxBatch =
          mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMAX).d[0];

      if (mMinBatchSize == 0) {
        mMinBatchSize = minBatch;
        mOptBatchSize = optBatch;
        mMaxBatchSize = maxBatch;
      } else if (minBatch != mMinBatchSize || optBatch != mOptBatchSize ||
                 maxBatch != mMaxBatchSize) {
        throw std::runtime_error(
            "Inconsistent batch profile across input tensors: '" +
            metadata.name + "' has [" + std::to_string(minBatch) + "," +
            std::to_string(optBatch) + "," + std::to_string(maxBatch) +
            "] but expected [" + std::to_string(mMinBatchSize) + "," +
            std::to_string(mOptBatchSize) + "," +
            std::to_string(mMaxBatchSize) + "]");
      }
    } else if (tensorType == TensorIOMode::kOUTPUT) {
      uint32_t outputLen = 1;
      for (int j = 1; j < tensorShape.nbDims; ++j) {
        outputLen *= tensorShape.d[j];
      }
      mOutputLengths.push_back(outputLen);
    } else {
      throw std::runtime_error(
          "Error, IO Tensor is neither an input or output!");
    }
  }
}

void Engine::enqueue(const uint64_t *input_ptrs, size_t num_inputs,
                     const uint64_t *output_ptrs, size_t num_outputs,
                     cudaStream_t stream, uint32_t batch_size) {
  if (batch_size < static_cast<uint32_t>(mMinBatchSize)) {
    throw std::runtime_error(
        "Batch size " + std::to_string(batch_size) +
        " is less than minimum: " + std::to_string(mMinBatchSize));
  }
  if (batch_size > static_cast<uint32_t>(mMaxBatchSize)) {
    throw std::runtime_error(
        "Batch size " + std::to_string(batch_size) +
        " is greater than maximum: " + std::to_string(mMaxBatchSize));
  }

  if (num_inputs != get_num_inputs()) {
    throw std::runtime_error(
        "Expected " + std::to_string(get_num_inputs()) +
        " input pointers, got " + std::to_string(num_inputs));
  }
  if (num_outputs != get_num_outputs()) {
    throw std::runtime_error(
        "Expected " + std::to_string(get_num_outputs()) +
        " output pointers, got " + std::to_string(num_outputs));
  }

  size_t inputIdx = 0;
  size_t outputIdx = 0;

  for (size_t i = 0; i < mTensorMetadata.size(); ++i) {
    const auto &meta = mTensorMetadata[i];

    if (meta.ioMode == TensorIOMode::kINPUT) {
      Dims inputDims = meta.dims;
      inputDims.d[0] = batch_size;
      if (!mContext->setInputShape(meta.name.c_str(), inputDims)) {
        throw std::runtime_error("Failed to set input shape for tensor: " +
                                 meta.name);
      }
      if (!mContext->setTensorAddress(meta.name.c_str(),
                                      reinterpret_cast<void *>(input_ptrs[inputIdx]))) {
        throw std::runtime_error("Unable to set tensor address for: " +
                                 meta.name);
      }
      inputIdx++;
    } else {
      if (!mContext->setTensorAddress(meta.name.c_str(),
                                      reinterpret_cast<void *>(output_ptrs[outputIdx]))) {
        throw std::runtime_error("Unable to set tensor address for: " +
                                 meta.name);
      }
      outputIdx++;
    }
  }

  if (!mContext->allInputDimensionsSpecified()) {
    throw std::runtime_error("Error, not all required dimensions specified.");
  }

  if (!mContext->enqueueV3(stream)) {
    throw std::runtime_error("Inference execution failed");
  }
}

void Engine::infer(const uint64_t *input_ptrs, size_t num_inputs,
                   const uint64_t *output_ptrs, size_t num_outputs,
                   uint64_t stream, uint32_t batch_size) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  enqueue(input_ptrs, num_inputs, output_ptrs, num_outputs, cuda_stream,
          batch_size);
  checkCudaErrorCode(cudaStreamSynchronize(cuda_stream));
}

void Engine::infer_async(const uint64_t *input_ptrs, size_t num_inputs,
                         const uint64_t *output_ptrs, size_t num_outputs,
                         uint64_t stream, uint32_t batch_size) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  enqueue(input_ptrs, num_inputs, output_ptrs, num_outputs, cuda_stream,
          batch_size);
}

rust::Vec<TensorInfo> Engine::get_input_dims() const {
  rust::Vec<TensorInfo> result;
  for (const auto &meta : mTensorMetadata) {
    if (meta.ioMode == TensorIOMode::kINPUT) {
      TensorInfo info;
      info.name = meta.name;
      resize(info.dims, 0);
      for (int j = 1; j < meta.dims.nbDims; ++j) {
        info.dims.push_back(static_cast<uint32_t>(meta.dims.d[j]));
      }
      info.dtype = toTensorDataType(meta.dataType);
      result.push_back(std::move(info));
    }
  }
  return result;
}

rust::Vec<TensorInfo> Engine::get_output_dims() const {
  rust::Vec<TensorInfo> result;
  for (const auto &meta : mTensorMetadata) {
    if (meta.ioMode == TensorIOMode::kOUTPUT) {
      TensorInfo info;
      info.name = meta.name;
      resize(info.dims, 0);
      for (int j = 1; j < meta.dims.nbDims; ++j) {
        info.dims.push_back(static_cast<uint32_t>(meta.dims.d[j]));
      }
      info.dtype = toTensorDataType(meta.dataType);
      result.push_back(std::move(info));
    }
  }
  return result;
}

rust::Vec<uint32_t> Engine::_get_batch_dims() const {
  rust::Vec<uint32_t> rv;
  rv.push_back(mMinBatchSize);
  rv.push_back(mOptBatchSize);
  rv.push_back(mMaxBatchSize);
  return rv;
}

uint32_t Engine::get_output_len() const {
  return mOutputLengths.empty() ? 0 : mOutputLengths[0];
}

size_t Engine::get_num_inputs() const {
  size_t count = 0;
  for (const auto &meta : mTensorMetadata) {
    if (meta.ioMode == TensorIOMode::kINPUT) {
      count++;
    }
  }
  return count;
}

size_t Engine::get_num_outputs() const {
  size_t count = 0;
  for (const auto &meta : mTensorMetadata) {
    if (meta.ioMode == TensorIOMode::kOUTPUT) {
      count++;
    }
  }
  return count;
}
