#include "engine.h"

#include <NvOnnxParser.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>

#include "libinfer/src/lib.rs.h"

using namespace nvinfer1;

// Implement our Rust friends.
std::unique_ptr<Engine> load_engine(const Options &options) {
  auto engine = std::make_unique<Engine>(options);
  engine->load();
  return engine;
}

// Throw an exception on error, which manifests as Result in Rust.
static inline void checkCudaErrorCode(cudaError_t code) {
  if (code != 0) {
    std::string errMsg =
        "CUDA operation failed with code: " + std::to_string(code) + "(" +
        cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
    throw std::runtime_error(errMsg);
  }
}

// Pretty dumb that we don't get a resize method in rust::Vec.
template <typename T> static void resize(rust::Vec<T> &v, size_t len) {
  v.reserve(len);
  while (v.size() < len) {
    v.push_back(T{});
  }
}

Engine::Engine(const Options &options)
    : kEnginePath(options.path), kDeviceIndex(options.device_index) {
  if (!spdlog::get("libinfer")) {
    spdlog::set_pattern("%+", spdlog::pattern_time_type::utc);
    spdlog::set_default_logger(spdlog::stderr_color_mt("libinfer"));

    const char *rustLogLevelEnvVar = "RUST_LOG";
    const char *rustLogLevel = getenv(rustLogLevelEnvVar);
    if (rustLogLevel == nullptr) {
      spdlog::set_level(spdlog::level::warn);
    } else {

      std::string rustLogLevelStr(rustLogLevel);
      if (rustLogLevelStr == "error") {
        spdlog::set_level(spdlog::level::err);
      } else if (rustLogLevelStr == "warn") {
        spdlog::set_level(spdlog::level::warn);
      } else if (rustLogLevelStr == "info") {
        spdlog::set_level(spdlog::level::info);
      } else if (rustLogLevelStr == "debug" || rustLogLevelStr == "trace") {
        spdlog::set_level(spdlog::level::debug);
      }
    }
  }
}

Engine::~Engine() {
  // Free the GPU memory
  for (auto &buffer : mBuffers) {
    checkCudaErrorCode(cudaFree(buffer));
  }

  mBuffers.clear();
}

void Engine::load() {
  // Read the serialized model from disk
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

  // Create a runtime to deserialize the engine file.
  mRuntime = std::unique_ptr<IRuntime>{createInferRuntime(mLogger)};
  if (!mRuntime) {
    throw std::runtime_error("Runtime not initialized");
  }

  // Set the device index
  const auto ret = cudaSetDevice(kDeviceIndex);
  if (ret != 0) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    const auto errMsg =
        "Unable to set GPU device index to: " + std::to_string(kDeviceIndex) +
        ". Note, your device has " + std::to_string(numGPUs) +
        " CUDA-capable GPU(s).";
    throw std::runtime_error(errMsg);
  }

  // Create an engine, a representation of the optimized model.
  mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
      mRuntime->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!mEngine) {
    throw std::runtime_error("Engine deserialization failed");
  }

  // The execution context contains all of the state associated with a
  // particular invocation
  mContext = std::unique_ptr<nvinfer1::IExecutionContext>(
      mEngine->createExecutionContext());
  if (!mContext) {
    throw std::runtime_error("Could not create execution context");
  }

  // Storage for holding the input and output buffers
  // This will be passed to TensorRT for inference
  mBuffers.resize(mEngine->getNbIOTensors());

  // Create a cuda stream
  cudaStream_t stream;
  checkCudaErrorCode(cudaStreamCreate(&stream));

  // Allocate GPU memory for input and output buffers
  mOutputLengths.clear();
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    mIOTensorNames.emplace_back(tensorName);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorShape = mEngine->getTensorShape(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);
    if (tensorType == TensorIOMode::kINPUT) {
      // Store the input dims for later use.
      mInputDims.emplace_back(tensorShape.d[1], tensorShape.d[2],
                              tensorShape.d[3]);
      switch (tensorDataType) {
      case DataType::kFLOAT:
        mInputDataType = InputDataType::FP32;
        mInputDataTypeSize = 4;
        break;
      case DataType::kUINT8:
        mInputDataType = InputDataType::UINT8;
        mInputDataTypeSize = 1;
        break;
      default:
        mInputDataType = InputDataType::FP32;
        mInputDataTypeSize = 4;
        break;
      }

      mMinBatchSize =
          mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMIN)
              .d[0];
      mOptBatchSize =
          mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kOPT)
              .d[0];
      mMaxBatchSize =
          mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMAX)
              .d[0];

      // Allocate as much memory for the largest supported batch size.
      checkCudaErrorCode(
          cudaMallocAsync(&mBuffers[i],
                          mMaxBatchSize * tensorShape.d[1] * tensorShape.d[2] *
                              tensorShape.d[3] * mInputDataTypeSize,
                          stream));

    } else if (tensorType == TensorIOMode::kOUTPUT) {
      // The binding is an output
      uint32_t outputLenFloat = 1;
      mOutputDims.push_back(tensorShape);

      for (int j = 1; j < tensorShape.nbDims; ++j) {
        // We ignore j = 0 because that is the batch size, and we will take that
        // into account when sizing the buffer.
        outputLenFloat *= tensorShape.d[j];
      }

      mOutputLengths.push_back(outputLenFloat);

      // Allocate as much memory for the largest supported batch size.
      checkCudaErrorCode(cudaMallocAsync(
          &mBuffers[i], outputLenFloat * mMaxBatchSize * sizeof(float),
          stream));
    } else {
      throw std::runtime_error(
          "Error, IO Tensor is neither an input or output!");
    }
  }

  // Synchronize and destroy the cuda stream
  checkCudaErrorCode(cudaStreamSynchronize(stream));
  checkCudaErrorCode(cudaStreamDestroy(stream));
}

rust::Vec<float> Engine::infer(const rust::Vec<uint8_t> &input) {
  // Create the cuda stream that will be used for inference
  cudaStream_t inferenceCudaStream;
  checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

  const auto &dims = mInputDims[0];

  // Check that the passed batch size can be handled.
  const int32_t calculatedBatchSize =
      input.size() / (dims.d[0] * dims.d[1] * dims.d[2] * mInputDataTypeSize);

  if (calculatedBatchSize < mMinBatchSize) {
    throw std::runtime_error("Input is less the minimum batch size: " +
                             std::to_string(calculatedBatchSize) + " > " +
                             std::to_string(mMinBatchSize));
  }

  if (calculatedBatchSize > mMaxBatchSize) {
    throw std::runtime_error("Input is greater than maximum batch size: " +
                             std::to_string(calculatedBatchSize) + " > " +
                             std::to_string(mMaxBatchSize));
  }

  // Check that vector has enough elements for full input.
  if (input.size() % (dims.d[0] * dims.d[1] * dims.d[2]) != 0) {
    throw std::runtime_error(
        "Input vector does not contain a whole number of batches");
  }

  // Define the batch size.
  nvinfer1::Dims4 inputDims = {calculatedBatchSize, dims.d[0], dims.d[1],
                               dims.d[2]};
  mContext->setInputShape(mIOTensorNames[0].c_str(), inputDims);

  checkCudaErrorCode(cudaMemcpyAsync(mBuffers[0], input.data(), input.size(),
                                     cudaMemcpyHostToDevice,
                                     inferenceCudaStream));

  // Ensure all dynamic bindings have been defined.
  if (!mContext->allInputDimensionsSpecified()) {
    throw std::runtime_error("Error, not all required dimensions specified.");
  }

  // Set the address of the input and output buffers
  for (size_t i = 0; i < mBuffers.size(); ++i) {
    bool status =
        mContext->setTensorAddress(mIOTensorNames[i].c_str(), mBuffers[i]);
    if (!status) {
      throw std::runtime_error("Unable to set tensor address in context");
    }
  }

  // Run inference.
  bool status = mContext->enqueueV3(inferenceCudaStream);
  if (!status) {
    throw std::runtime_error("enqueue failed");
  }

  const auto outputLen = calculatedBatchSize * mOutputLengths[0];
  rust::Vec<float> output;
  resize(output, outputLen);
  checkCudaErrorCode(cudaMemcpyAsync(
      output.data(), static_cast<char *>(mBuffers[1]),
      outputLen * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));

  // Synchronize the cuda stream
  checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
  checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

  return output;
}
