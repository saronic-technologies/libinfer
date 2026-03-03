#include "engine.h"

#include <NvInferRuntimeBase.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "libinfer/src/lib.rs.h"

#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>


using namespace nvinfer1;

// Helper function to convert from our enum to TensorRT's enum.
TensorDataType toTensorDataType(DataType dt) {
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

// Helper function to get the size in bytes of a data type.
size_t getDataTypeSize(TensorDataType dt) {
  switch (dt) {
  case TensorDataType::FP32:
    return 4;
  case TensorDataType::UINT8:
    return 1;
  case TensorDataType::INT64:
    return 8;
  case TensorDataType::BOOL:
    return 1;
  }
}

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
  // Free the GPU memory. Do not use checkCudaErrorCode here. Destructors
  // must not throw. If cudaFree fails (e.g. context already destroyed) we
  // log and continue rather than calling std::terminate via an exception
  // during stack unwinding.
  for (auto &buffer : mBuffers) {
    auto err = cudaFree(buffer);
    if (err != cudaSuccess) {
      spdlog::error("cudaFree failed in ~Engine: {} ({})",
                    cudaGetErrorName(err), cudaGetErrorString(err));
    }
  }

  if (mInferenceCudaStream) {
    auto err = cudaStreamDestroy(mInferenceCudaStream);
    if (err != cudaSuccess) {
      spdlog::error("cudaStreamDestroy failed in ~Engine: {} ({})",
                    cudaGetErrorName(err), cudaGetErrorString(err));
    }
  }

  mBuffers.clear();
}

void Engine::load() {
  // Initialize plugins
  initLibNvInferPlugins(&mLogger, "");


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

  // Create the cuda stream that will be used for inference
  checkCudaErrorCode(cudaStreamCreate(&mInferenceCudaStream));

  // Allocate GPU memory for input and output buffers.
  mTensorMetadata.clear();
  mTensorMetadata.reserve(mEngine->getNbIOTensors());

  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    mIOTensorNames.emplace_back(tensorName);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorShape = mEngine->getTensorShape(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);

    TensorMetadata metadata;
    metadata.name = std::string(tensorName);
    metadata.ioMode = tensorType;
    metadata.dataType = tensorDataType;
    metadata.dataTypeSize = getDataTypeSize(toTensorDataType(tensorDataType));
    metadata.dims = tensorShape;

    if (tensorType == TensorIOMode::kINPUT) {
      mInputDims.push_back(tensorShape);

      // Query per-input profile shapes
      metadata.minShape = mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMIN);
      metadata.optShape = mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kOPT);
      metadata.maxShape = mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMAX);

      // Detect dynamic dimensions and precompute inference fields
      metadata.hasDynamicShape = false;
      metadata.dynamicDimIndex = -1;
      metadata.staticByteCount = metadata.dataTypeSize;

      for (int j = 0; j < tensorShape.nbDims; ++j) {
        if (tensorShape.d[j] == -1) {
          if (metadata.hasDynamicShape) {
            throw std::runtime_error(
                "Input '" + metadata.name + "' has multiple dynamic dimensions; "
                "exactly 1 is supported for automatic shape inference");
          }
          metadata.hasDynamicShape = true;
          metadata.dynamicDimIndex = j;
          // Don't multiply this dim into staticByteCount
        } else {
          metadata.staticByteCount *= tensorShape.d[j];
        }
      }

      // Allocate buffer using this input's own max shape
      size_t inputBufferBytes = metadata.dataTypeSize;
      for (int j = 0; j < metadata.maxShape.nbDims; ++j) {
        inputBufferBytes *= metadata.maxShape.d[j];
      }

      checkCudaErrorCode(
        cudaMallocAsync(&mBuffers[i], inputBufferBytes, mInferenceCudaStream)
      );
    } else if (tensorType == TensorIOMode::kOUTPUT) {
      mOutputDims.push_back(tensorShape);
      // Output buffers allocated after the loop via shape propagation
    } else {
      throw std::runtime_error(
          "Error, IO Tensor is neither an input or output!");
    }

    // Push metadata after all fields are populated
    mTensorMetadata.push_back(std::move(metadata));
  }

  // Allocate output buffers by setting all inputs to their max shapes
  // and querying TensorRT for the resulting output shapes.
  for (const auto &meta : mTensorMetadata) {
    if (meta.ioMode == TensorIOMode::kINPUT) {
      mContext->setInputShape(meta.name.c_str(), meta.maxShape);
    }
  }

  for (int i = 0; i < static_cast<int>(mTensorMetadata.size()); ++i) {
    const auto &meta = mTensorMetadata[i];
    if (meta.ioMode != TensorIOMode::kOUTPUT) continue;

    nvinfer1::Dims maxOutputShape = mContext->getTensorShape(meta.name.c_str());

    size_t outputBufferBytes = meta.dataTypeSize;
    for (int j = 0; j < maxOutputShape.nbDims; ++j) {
      outputBufferBytes *= maxOutputShape.d[j];
    }

    checkCudaErrorCode(
      cudaMallocAsync(&mBuffers[i], outputBufferBytes, mInferenceCudaStream)
    );
  }

  // Ensure all async allocations are complete before returning.
  checkCudaErrorCode(cudaStreamSynchronize(mInferenceCudaStream));
}

rust::Vec<OutputTensor> Engine::infer(const rust::Vec<InputTensor> &input) {
  // Create a map from tensor name to input data for easy lookup
  std::unordered_map<std::string, const InputTensor*> inputMap;
  for (const auto &tensorInput : input) {
    inputMap[std::string(tensorInput.name)] = &tensorInput;
  }

  // Process each input tensor using cached metadata (per-input shape resolution)
  for (int i = 0; i < static_cast<int>(mTensorMetadata.size()); ++i) {
    const auto &metadata = mTensorMetadata[i];

    if (metadata.ioMode != TensorIOMode::kINPUT) {
      continue;
    }

    auto it = inputMap.find(metadata.name);
    if (it == inputMap.end()) {
      throw std::runtime_error("Missing input tensor: " + metadata.name);
    }

    const auto &tensorInput = *it->second;
    nvinfer1::Dims actualShape = metadata.dims;

    if (metadata.hasDynamicShape) {
      // Resolve the single dynamic dimension via precomputed staticByteCount
      if (tensorInput.data.size() % metadata.staticByteCount != 0) {
        throw std::runtime_error(
            "Input tensor '" + metadata.name + "' data size (" +
            std::to_string(tensorInput.data.size()) +
            " bytes) is not divisible by static element size (" +
            std::to_string(metadata.staticByteCount) + " bytes)");
      }

      int32_t dynamicDimValue = static_cast<int32_t>(
          tensorInput.data.size() / metadata.staticByteCount);

      if (dynamicDimValue < metadata.minShape.d[metadata.dynamicDimIndex]) {
        throw std::runtime_error(
            "Input '" + metadata.name + "' dynamic dim[" +
            std::to_string(metadata.dynamicDimIndex) + "] = " +
            std::to_string(dynamicDimValue) + " is less than min profile value " +
            std::to_string(metadata.minShape.d[metadata.dynamicDimIndex]));
      }
      if (dynamicDimValue > metadata.maxShape.d[metadata.dynamicDimIndex]) {
        throw std::runtime_error(
            "Input '" + metadata.name + "' dynamic dim[" +
            std::to_string(metadata.dynamicDimIndex) + "] = " +
            std::to_string(dynamicDimValue) + " exceeds max profile value " +
            std::to_string(metadata.maxShape.d[metadata.dynamicDimIndex]));
      }

      actualShape.d[metadata.dynamicDimIndex] = dynamicDimValue;
    } else {
      // Static input: validate exact size
      if (tensorInput.data.size() != metadata.staticByteCount) {
        throw std::runtime_error(
            "Input tensor '" + metadata.name + "' data size (" +
            std::to_string(tensorInput.data.size()) +
            " bytes) does not match expected size (" +
            std::to_string(metadata.staticByteCount) + " bytes)");
      }
    }

    bool shapeStatus = mContext->setInputShape(metadata.name.c_str(), actualShape);
    if (!shapeStatus) {
      throw std::runtime_error("Failed to set input shape for tensor: " + metadata.name);
    }

    checkCudaErrorCode(cudaMemcpyAsync(mBuffers[i], tensorInput.data.data(),
                                       tensorInput.data.size(),
                                       cudaMemcpyHostToDevice, mInferenceCudaStream));
  }

  // No pre-enqueue sync needed: CUDA stream ordering guarantees H2D copies
  // complete before enqueueV3 kernels begin. See README.md for details.

  // Ensure all dynamic bindings have been defined
  if (!mContext->allInputDimensionsSpecified()) {
    throw std::runtime_error("Error, not all required dimensions specified.");
  }
  
  // Set the address of all input and output buffers using cached names
  for (int i = 0; i < static_cast<int>(mTensorMetadata.size()); ++i) {
    bool status = mContext->setTensorAddress(mTensorMetadata[i].name.c_str(), mBuffers[i]);
    if (!status) {
      throw std::runtime_error("Unable to set tensor address for: " + mTensorMetadata[i].name);
    }
  }
  
  // Run inference
  bool status = mContext->enqueueV3(mInferenceCudaStream);
  if (!status) {
    throw std::runtime_error("Inference execution failed");
  }
  
  // Collect output tensors using dynamically-inferred shapes
  rust::Vec<OutputTensor> outputs;

  for (int i = 0; i < static_cast<int>(mTensorMetadata.size()); ++i) {
    const auto &metadata = mTensorMetadata[i];

    if (metadata.ioMode != TensorIOMode::kOUTPUT) {
      continue;
    }

    // Query the actual output shape from TensorRT (reflects current input shapes)
    nvinfer1::Dims outputShape = mContext->getTensorShape(metadata.name.c_str());

    size_t copySize = metadata.dataTypeSize;
    for (int d = 0; d < outputShape.nbDims; ++d) {
      copySize *= outputShape.d[d];
    }

    OutputTensor output;
    output.name = metadata.name;
    output.dtype = toTensorDataType(metadata.dataType);
    output.data = new_output_buffer(copySize);

    checkCudaErrorCode(cudaMemcpyAsync(output.data.data(),
                                       static_cast<char *>(mBuffers[i]),
                                       copySize,
                                       cudaMemcpyDeviceToHost, mInferenceCudaStream));

    outputs.push_back(std::move(output));
  }

  checkCudaErrorCode(cudaStreamSynchronize(mInferenceCudaStream));

  return outputs;
}

size_t Engine::get_num_inputs() const {
  size_t count = 0;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode == TensorIOMode::kINPUT) {
      count++;
    }
  }
  return count;
}

size_t Engine::get_num_outputs() const {
  size_t count = 0;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode == TensorIOMode::kOUTPUT) {
      count++;
    }
  }
  return count;
}

rust::Vec<TensorInfo> Engine::get_input_dims() const {
  rust::Vec<TensorInfo> result;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode == TensorIOMode::kINPUT) {
      TensorInfo info;
      info.name = metadata.name;
      resize(info.dims, 0);
      // Skip batch dimension (index 0)
      for (int j = 1; j < metadata.dims.nbDims; ++j) {
        info.dims.push_back(static_cast<uint32_t>(metadata.dims.d[j]));
      }
      info.dtype = toTensorDataType(metadata.dataType);
      result.push_back(std::move(info));
    }
  }
  return result;
}

rust::Vec<TensorInfo> Engine::get_output_dims() const {
  rust::Vec<TensorInfo> result;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode == TensorIOMode::kOUTPUT) {
      TensorInfo info;
      info.name = metadata.name;
      resize(info.dims, 0);
      // Skip batch dimension (index 0)
      for (int j = 1; j < metadata.dims.nbDims; ++j) {
        info.dims.push_back(static_cast<uint32_t>(metadata.dims.d[j]));
      }
      info.dtype = toTensorDataType(metadata.dataType);
      result.push_back(std::move(info));
    }
  }
  return result;
}

rust::Vec<InputShapeProfile> Engine::_get_input_shape_profiles() const {
  rust::Vec<InputShapeProfile> result;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode != TensorIOMode::kINPUT) continue;

    InputShapeProfile profile;
    profile.name = metadata.name;
    profile.has_dynamic_shape = metadata.hasDynamicShape;

    resize(profile.min_shape, 0);
    resize(profile.opt_shape, 0);
    resize(profile.max_shape, 0);

    for (int j = 0; j < metadata.minShape.nbDims; ++j) {
      profile.min_shape.push_back(static_cast<int32_t>(metadata.minShape.d[j]));
    }
    for (int j = 0; j < metadata.optShape.nbDims; ++j) {
      profile.opt_shape.push_back(static_cast<int32_t>(metadata.optShape.d[j]));
    }
    for (int j = 0; j < metadata.maxShape.nbDims; ++j) {
      profile.max_shape.push_back(static_cast<int32_t>(metadata.maxShape.d[j]));
    }

    result.push_back(std::move(profile));
  }
  return result;
}
