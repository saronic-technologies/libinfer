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
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    mIOTensorNames.emplace_back(tensorName);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorShape = mEngine->getTensorShape(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);
    if (tensorType == TensorIOMode::kINPUT) {
      // Store the input dims for later use.
      mInputDims.push_back(tensorShape);
      const size_t inputDataTypeSize = getDataTypeSize(toTensorDataType(tensorDataType));

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
      const size_t outputDataTypeSize = getDataTypeSize(toTensorDataType(tensorDataType));

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
  spdlog::debug("Engine::infer() called with {} input tensors", input.size());
  
  // Create a map from tensor name to input data for easy lookup
  std::unordered_map<std::string, const InputTensor*> inputMap;
  for (const auto &tensorInput : input) {
    spdlog::debug("Processing input tensor '{}' with {} bytes", 
                  std::string(tensorInput.name), tensorInput.data.size());
    inputMap[std::string(tensorInput.name)] = &tensorInput;
  }

  // Track the batch size (should be consistent across all inputs)
  int32_t batchSize = -1;
  
  spdlog::debug("Processing {} IO tensors from engine", mEngine->getNbIOTensors());
  
  // Process each input tensor
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);
    const auto tensorDataTypeSize = getDataTypeSize(toTensorDataType(tensorDataType));

    spdlog::debug("Processing IO tensor {}: '{}', type={}, datatype={}",
                  i, std::string(tensorName), 
                  (tensorType == TensorIOMode::kINPUT ? "INPUT" : "OUTPUT"),
                  tensorDataTypeSize);

    if (tensorType != TensorIOMode::kINPUT) {
      continue; // If not a tensor input skip
    }
    
    // Find the corresponding input data
    auto it = inputMap.find(tensorName);
    if (it == inputMap.end()) {
      spdlog::error("Missing input tensor: '{}'", std::string(tensorName));
      throw std::runtime_error("Missing input tensor: " + std::string(tensorName));
    }
    
    const auto &tensorInput = *it->second;
    spdlog::debug("Found input tensor '{}' with {} bytes", std::string(tensorInput.name), tensorInput.data.size());
    
    const auto &dims = mInputDims[i]; // Assuming mInputDims indexed by tensor order
    spdlog::debug("Input dims for tensor {}: nbDims={}", i, dims.nbDims);
    
    // Calculate expected tensor size (excluding batch dimension)
    size_t tensorSize = 1;
    for (int d = 1; d < dims.nbDims; ++d) {
      spdlog::debug("Tensor dimension {}: {}", d, dims.d[d]);
      tensorSize *= dims.d[d];
    }
    spdlog::debug("Base tensor size (before data type): {}", tensorSize);
    tensorSize *= tensorDataTypeSize; // Account for data type size
    spdlog::debug("Tensor size after data type ({}): {}", tensorDataTypeSize, tensorSize);
    
    // Calculate batch size from input data
    int32_t currentBatchSize = tensorInput.data.size() / tensorSize;
    spdlog::debug("Calculated batch size: {} (input.size={}, tensorSize={})", 
                  currentBatchSize, tensorInput.data.size(), tensorSize);
    
    if (batchSize == -1) {
      batchSize = currentBatchSize;
      spdlog::debug("Set batch size to: {} (min={}, max={})", batchSize, mMinBatchSize, mMaxBatchSize);
      
      // Validate batch size constraints
      if (batchSize < mMinBatchSize) {
        spdlog::error("Input batch size {} is less than minimum: {}", batchSize, mMinBatchSize);
        throw std::runtime_error("Input batch size " + std::to_string(batchSize) + 
                               " is less than minimum: " + std::to_string(mMinBatchSize));
      }
      if (batchSize > mMaxBatchSize) {
        spdlog::error("Input batch size {} is greater than maximum: {}", batchSize, mMaxBatchSize);
        throw std::runtime_error("Input batch size " + std::to_string(batchSize) + 
                               " is greater than maximum: " + std::to_string(mMaxBatchSize));
      }
    } else if (currentBatchSize != batchSize) {
      spdlog::error("Inconsistent batch sizes: {} vs {}", currentBatchSize, batchSize);
      throw std::runtime_error("Inconsistent batch sizes across input tensors");
    }
    
    // Validate input tensor size
    if (tensorInput.data.size() % tensorSize != 0) {
      spdlog::error("Input tensor '{}' size validation failed: {} bytes, expected multiple of {}", 
                    std::string(tensorName), tensorInput.data.size(), tensorSize);
      throw std::runtime_error("Input tensor '" + std::string(tensorName) + 
                              "' does not contain whole number of batches");
    }
    
    // Set input shape with batch dimension
    nvinfer1::Dims inputDims = dims;
    inputDims.d[0] = batchSize;
    spdlog::debug("Setting input shape for '{}': batch_dim={}", std::string(tensorName), batchSize);
    bool shapeStatus = mContext->setInputShape(tensorName, inputDims);
    if (!shapeStatus) {
      spdlog::error("Failed to set input shape for tensor '{}'", std::string(tensorName));
      throw std::runtime_error("Failed to set input shape for tensor: " + std::string(tensorName));
    }
    
    // Copy input data to GPU buffer
    spdlog::debug("Copying {} bytes to GPU buffer for tensor '{}'", tensorInput.data.size(), std::string(tensorName));
    checkCudaErrorCode(cudaMemcpyAsync(mBuffers[i], tensorInput.data.data(), 
                                      tensorInput.data.size(),
                                      cudaMemcpyHostToDevice, mInferenceCudaStream));
  }

  spdlog::debug("Synchronizing CUDA stream after input copy");
  checkCudaErrorCode(cudaStreamSynchronize(mInferenceCudaStream));
  
  // Ensure all dynamic bindings have been defined
  spdlog::debug("Checking if all input dimensions are specified");
  if (!mContext->allInputDimensionsSpecified()) {
    spdlog::error("Not all required input dimensions specified");
    throw std::runtime_error("Error, not all required dimensions specified.");
  }
  
  // Set the address of all input and output buffers
  spdlog::debug("Setting tensor addresses for {} IO tensors", mEngine->getNbIOTensors());
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    spdlog::debug("Setting tensor address for '{}' at buffer index {}", std::string(tensorName), i);
    bool status = mContext->setTensorAddress(tensorName, mBuffers[i]);
    if (!status) {
      spdlog::error("Failed to set tensor address for '{}'", std::string(tensorName));
      throw std::runtime_error("Unable to set tensor address for: " + std::string(tensorName));
    }
  }
  
  // Run inference
  spdlog::debug("Starting inference execution (enqueueV3)");
  bool status = mContext->enqueueV3(mInferenceCudaStream);
  if (!status) {
    spdlog::error("Inference execution failed");
    throw std::runtime_error("Inference execution failed");
  }
  spdlog::debug("Inference execution completed successfully");
  
  // Collect output tensors
  spdlog::debug("Collecting output tensors");
  rust::Vec<OutputTensor> outputs;
  
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);
    const auto tensorDataTypeSize = getDataTypeSize(toTensorDataType(tensorDataType));

    if (tensorType != TensorIOMode::kOUTPUT) {
      continue; // skip if not tensor output
    }
    
    spdlog::debug("Processing output tensor {}: '{}', datatype_size={}", 
                  i, std::string(tensorName), tensorDataTypeSize);
    
    // Find the output length for this tensor
    size_t outputIdx = 0; // Need to map from tensor index to output index
    for (int j = 0; j < i; ++j) {
      if (mEngine->getTensorIOMode(mEngine->getIOTensorName(j)) == TensorIOMode::kOUTPUT) {
        outputIdx++;
      }
    }
    
    const auto outputLen = batchSize * mOutputLengths[outputIdx];
    spdlog::debug("Output tensor '{}': outputIdx={}, base_length={}, batch_size={}, total_elements={}", 
                  std::string(tensorName), outputIdx, mOutputLengths[outputIdx], batchSize, outputLen);
    
    // Create output tensor
    OutputTensor output;
    size_t copySize = outputLen * tensorDataTypeSize;
    output.name = std::string(tensorName);
    spdlog::debug("Resizing output data vector to {} bytes", copySize);
    resize(output.data, copySize);
    
    // Copy data from GPU buffer
    spdlog::debug("Copying {} bytes from GPU buffer for tensor '{}'", copySize, std::string(tensorName));
    checkCudaErrorCode(cudaMemcpyAsync(output.data.data(), 
                                      static_cast<char*>(mBuffers[i]),
                                      copySize, 
                                      cudaMemcpyDeviceToHost, mInferenceCudaStream));
    
    outputs.push_back(std::move(output));
  }
  
  spdlog::debug("Synchronizing CUDA stream after output copy");
  checkCudaErrorCode(cudaStreamSynchronize(mInferenceCudaStream));
  
  spdlog::debug("Engine::infer() completed successfully with {} outputs", outputs.size());
  return outputs;
}

size_t Engine::get_num_inputs() const {
  size_t count = 0;
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    if (tensorType == TensorIOMode::kINPUT) {
      count++;
    }
  }
  return count;
}

size_t Engine::get_num_outputs() const {
  size_t count = 0;
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    if (tensorType == TensorIOMode::kOUTPUT) {
      count++;
    }
  }
  return count;
}

rust::Vec<TensorInfo> Engine::get_input_dims() const {
  rust::Vec<TensorInfo> result;
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);
    if (tensorType == TensorIOMode::kINPUT) {
      const auto dims = mEngine->getTensorShape(tensorName);
      TensorInfo info;
      info.name = std::string(tensorName);
      resize(info.dims, 0);
      // Skip batch dimension (index 0)
      for (int j = 1; j < dims.nbDims; ++j) {
        info.dims.push_back(static_cast<uint32_t>(dims.d[j]));
      }

      info.dtype = toTensorDataType(tensorDataType); 

      result.push_back(std::move(info));
    }
  }
  return result;
}

rust::Vec<TensorInfo> Engine::get_output_dims() const {
  rust::Vec<TensorInfo> result;
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);
    if (tensorType == TensorIOMode::kOUTPUT) {
      const auto dims = mEngine->getTensorShape(tensorName);
      TensorInfo info;
      info.name = std::string(tensorName);
      resize(info.dims, 0);
      // Skip batch dimension (index 0)
      for (int j = 1; j < dims.nbDims; ++j) {
        info.dims.push_back(static_cast<uint32_t>(dims.d[j]));
      }

      info.dtype = toTensorDataType(tensorDataType); 

      result.push_back(std::move(info));
    }
  }
  return result;
}
