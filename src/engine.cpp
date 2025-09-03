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
  // Free the GPU memory
  for (auto &buffer : mBuffers) {
    checkCudaErrorCode(cudaFree(buffer));
  }

  checkCudaErrorCode(cudaStreamDestroy(mInferenceCudaStream));

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

  // Create a cuda stream
  cudaStream_t stream;
  checkCudaErrorCode(cudaStreamCreate(&stream));

  // Allocate GPU memory for input and output buffers
  mOutputLengths.clear();
  mTensorMetadata.clear();
  mTensorMetadata.reserve(mEngine->getNbIOTensors());
  
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    mIOTensorNames.emplace_back(tensorName);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorShape = mEngine->getTensorShape(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);
    
    // Store tensor metadata to avoid repeated TensorRT queries during inference
    TensorMetadata metadata;
    metadata.name = std::string(tensorName);
    metadata.ioMode = tensorType;
    metadata.dataType = tensorDataType;
    metadata.dataTypeSize = getDataTypeSize(toTensorDataType(tensorDataType));
    metadata.dims = tensorShape;
    mTensorMetadata.push_back(std::move(metadata));
    if (tensorType == TensorIOMode::kINPUT) {
      // Store the input dims for later use.
      mInputDims.push_back(tensorShape);
      const size_t inputDataTypeSize = metadata.dataTypeSize;

      mMinBatchSize =
          mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMIN)
              .d[0];
      mOptBatchSize =
          mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kOPT)
              .d[0];
      mMaxBatchSize =
          mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMAX)
              .d[0];

      // multiply tensorShape dimensions except for the batch dimension
      uint32_t inputLen = 1;
      for (int j = 1; j < tensorShape.nbDims; ++j) {
        // We ignore j = 0 because that is the batch size, and we will take that
        // into account when sizing the buffer.
        inputLen *= tensorShape.d[j];
      }

      inputLen *= inputDataTypeSize; // Account for data type size
      inputLen *= mMaxBatchSize; // Account for max batch size

      // Allocate as much memory for the largest supported batch size.
      checkCudaErrorCode(
        cudaMallocAsync(&mBuffers[i],
          inputLen,
          stream
        )
      );
    } else if (tensorType == TensorIOMode::kOUTPUT) {
      const size_t outputDataTypeSize = metadata.dataTypeSize;

      // The binding is an output
      uint32_t outputLen = 1;
      mOutputDims.push_back(tensorShape);

      for (int j = 1; j < tensorShape.nbDims; ++j) {
        // We ignore j = 0 because that is the batch size, and we will take that
        // into account when sizing the buffer.
        outputLen *= tensorShape.d[j];
      }

      mOutputLengths.push_back(outputLen);

      outputLen *= outputDataTypeSize; // Account for data type size
      outputLen *= mMaxBatchSize; // Account for max batch size

      // Allocate as much memory for the largest supported batch size.
      checkCudaErrorCode(cudaMallocAsync(
          &mBuffers[i], 
          outputLen,
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

rust::Vec<OutputTensor> Engine::infer(const rust::Vec<InputTensor> &input) {
  // Create a map from tensor name to input data for easy lookup
  std::unordered_map<std::string, const InputTensor*> inputMap;
  for (const auto &tensorInput : input) {
    inputMap[std::string(tensorInput.name)] = &tensorInput;
  }

  // Track the batch size (should be consistent across all inputs)
  int32_t batchSize = -1;
  
  // Process each input tensor using cached metadata
  for (int i = 0; i < static_cast<int>(mTensorMetadata.size()); ++i) {
    const auto &metadata = mTensorMetadata[i];
    
    if (metadata.ioMode != TensorIOMode::kINPUT) {
      continue; // If not a tensor input skip
    }
    
    // Find the corresponding input data
    auto it = inputMap.find(metadata.name);
    if (it == inputMap.end()) {
      throw std::runtime_error("Missing input tensor: " + metadata.name);
    }
    
    const auto &tensorInput = *it->second;
    const auto &dims = metadata.dims;
    
    // Calculate expected tensor size (excluding batch dimension)
    size_t tensorSize = 1;
    for (int d = 1; d < dims.nbDims; ++d) {
      tensorSize *= dims.d[d];
    }
    tensorSize *= metadata.dataTypeSize; // Account for data type size
    
    // Calculate batch size from input data
    int32_t currentBatchSize = tensorInput.data.size() / tensorSize;
    
    if (batchSize == -1) {
      batchSize = currentBatchSize;
      
      // Validate batch size constraints
      if (batchSize < mMinBatchSize) {
        throw std::runtime_error("Input batch size " + std::to_string(batchSize) + 
                               " is less than minimum: " + std::to_string(mMinBatchSize));
      }
      if (batchSize > mMaxBatchSize) {
        throw std::runtime_error("Input batch size " + std::to_string(batchSize) + 
                               " is greater than maximum: " + std::to_string(mMaxBatchSize));
      }
    } else if (currentBatchSize != batchSize) {
      throw std::runtime_error("Inconsistent batch sizes across input tensors");
    }
    
    // Validate input tensor size
    if (tensorInput.data.size() % tensorSize != 0) {
      throw std::runtime_error("Input tensor '" + metadata.name + 
                              "' does not contain whole number of batches");
    }
    
    // Set input shape with batch dimension
    nvinfer1::Dims inputDims = dims;
    inputDims.d[0] = batchSize;
    bool shapeStatus = mContext->setInputShape(metadata.name.c_str(), inputDims);
    if (!shapeStatus) {
      throw std::runtime_error("Failed to set input shape for tensor: " + metadata.name);
    }
    
    // Copy input data to GPU buffer
    checkCudaErrorCode(cudaMemcpyAsync(mBuffers[i], tensorInput.data.data(), 
                                      tensorInput.data.size(),
                                      cudaMemcpyHostToDevice, mInferenceCudaStream));
  }

  checkCudaErrorCode(cudaStreamSynchronize(mInferenceCudaStream));
  
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
  
  // Collect output tensors
  rust::Vec<OutputTensor> outputs;
  
  for (int i = 0; i < static_cast<int>(mTensorMetadata.size()); ++i) {
    const auto &metadata = mTensorMetadata[i];
    
    if (metadata.ioMode != TensorIOMode::kOUTPUT) {
      continue; // skip if not tensor output
    }
    
    // Find the output length for this tensor
    size_t outputIdx = 0; // Need to map from tensor index to output index
    for (int j = 0; j < i; ++j) {
      if (mTensorMetadata[j].ioMode == TensorIOMode::kOUTPUT) {
        outputIdx++;
      }
    }
    
    const auto outputLen = batchSize * mOutputLengths[outputIdx];
    
    // Create output tensor
    OutputTensor output;
    size_t copySize = outputLen * metadata.dataTypeSize;
    output.name = metadata.name;
    output.dtype = toTensorDataType(metadata.dataType);
    resize(output.data, copySize);
    
    // Copy data from GPU buffer
    checkCudaErrorCode(cudaMemcpyAsync(output.data.data(), 
                                      static_cast<char*>(mBuffers[i]),
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
