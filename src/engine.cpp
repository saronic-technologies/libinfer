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
  mTensorMetadata.clear();
  mTensorMetadata.reserve(mEngine->getNbIOTensors());
  
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorShape = mEngine->getTensorShape(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);

    if (tensorType != TensorIOMode::kINPUT && tensorType != TensorIOMode::kOUTPUT) {
      throw std::runtime_error("Error, IO Tensor is neither an input or output!");
    }
    
    // Store tensor metadata to avoid repeated TensorRT queries during inference
    TensorMetadata metadata;
    metadata.name = std::string(tensorName);
    metadata.ioMode = tensorType;
    metadata.dataType = tensorDataType;
    metadata.dataTypeSize = getDataTypeSize(toTensorDataType(tensorDataType));
    metadata.shape = tensorShape;
    metadata.minShape = mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMIN);
    metadata.optShape = mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kOPT);
    metadata.maxShape = mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMAX);

    size_t nonDynamicSize = 1;
    for (int j = 0; j < tensorShape.nbDims; ++j) {
      if (tensorShape.d[j] == -1) {
        nonDynamicSize *= 1; // Treat dynamic dimensions as size 1
      } else {
        nonDynamicSize *= tensorShape.d[j];
      }
    }
    metadata.nonDynamicSize = nonDynamicSize * metadata.dataTypeSize;

    // find the size of the input buffer for max shape
    // multiply tensorShape dimensions for max shape by data type size
    uint32_t inputLen = metadata.dataTypeSize;
    for (int j = 0; j < tensorShape.nbDims; ++j) {
      inputLen *= metadata.maxShape.d[j];
    }

    // Allocate as much memory for the largest supported batch size.
    checkCudaErrorCode(
      cudaMallocAsync(&mBuffers[i],
        inputLen,
        stream
      )
    );
    
    mTensorMetadata.push_back(std::move(metadata));
  }

  // Synchronize and destroy the cuda stream
  checkCudaErrorCode(cudaStreamSynchronize(stream));
  checkCudaErrorCode(cudaStreamDestroy(stream));
}

rust::Vec<TensorInstance> Engine::infer(const rust::Vec<TensorInstance> &input) {
  // Create a map from tensor name to input data for easy lookup
  std::unordered_map<std::string, const TensorInstance*> inputMap;
  for (const auto &tensorInput : input) {
    inputMap[std::string(tensorInput.name)] = &tensorInput;
  }
  
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

    // validate input shape is valid with input size 
    if (static_cast<int32_t>(tensorInput.shape.size()) != metadata.shape.nbDims) {
      throw std::runtime_error("Input tensor '" + metadata.name + 
                              "' has incorrect number of dimensions: " + 
                              std::to_string(tensorInput.shape.size()) + 
                              ", expected: " + std::to_string(metadata.shape.nbDims));
    }

    size_t inputShapeSize = 1;
    for (size_t j = 0; j < tensorInput.shape.size(); ++j) {
      if (tensorInput.shape[j] < 1) {
        throw std::runtime_error(
                                "Invalid dimension size in input tensor '" + metadata.name + "'" + 
                                " at index " + std::to_string(j) + ": " + std::to_string(tensorInput.shape[j])
        );
      }

      if (tensorInput.shape[j] < metadata.minShape.d[j]) {
        throw std::runtime_error("Input tensor '" + metadata.name + 
                                "' dimension " + std::to_string(j) + 
                                " is smaller than minimum: " + 
                                std::to_string(tensorInput.shape[j]) + 
                                " < " + std::to_string(metadata.minShape.d[j]));
      }

      if (tensorInput.shape[j] > metadata.maxShape.d[j]) {
        throw std::runtime_error("Input tensor '" + metadata.name + 
                                "' dimension " + std::to_string(j) + 
                                " is larger than maximum: " + 
                                std::to_string(tensorInput.shape[j]) + 
                                " > " + std::to_string(metadata.maxShape.d[j]));
      }
      
      inputShapeSize *= tensorInput.shape[j];
    }

    inputShapeSize *= metadata.dataTypeSize; // account for data type size
    if (inputShapeSize != tensorInput.data.size()) {
      throw std::runtime_error("Input tensor '" + metadata.name + 
                              "' has size mismatch calculated bytes from input tensor shape: " + 
                              std::to_string(inputShapeSize) + 
                              " != " + std::to_string(tensorInput.data.size()));
    }
    
    // Validate input tensor size
    if (tensorInput.data.size() % metadata.nonDynamicSize != 0) {
      throw std::runtime_error("Input tensor '" + metadata.name + 
                              "' does not contain whole number of non-dynamic elements");
    }
    
    // set input shape from tensor input shape
    nvinfer1:: Dims inputDims = nvinfer1::Dims{};
    inputDims.nbDims = static_cast<int32_t>(tensorInput.shape.size());
    for (size_t j = 0; j < tensorInput.shape.size(); ++j) {
      inputDims.d[j] = tensorInput.shape[j];
    }

    bool shapeStatus = mContext->setInputShape(metadata.name.c_str(), inputDims);
    if (!shapeStatus) {
      throw std::runtime_error("Failed to set input shape for tensor: " + metadata.name);
    }
    
    // Copy input data to GPU buffer
    checkCudaErrorCode(cudaMemcpyAsync(mBuffers[i], tensorInput.data.data(), 
                                      tensorInput.data.size(),
                                      cudaMemcpyHostToDevice, mInferenceCudaStream));
  }
  
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

  // Get output tensor names from metadata
  char const** outputNames;
  int32_t numNames = 0;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode == TensorIOMode::kOUTPUT) {
      numNames++;
    }
  }

  outputNames = new char const*[numNames];
  int32_t outputIdx = 0;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode == TensorIOMode::kOUTPUT) {
      outputNames[outputIdx] = metadata.name.c_str();
      outputIdx++;
    }
  }

  // verify output shapes
  if (!mContext->inferShapes(numNames, outputNames)) {
    throw std::runtime_error("Shape inference failed");
  }
  
  // Run inference
  bool status = mContext->enqueueV3(mInferenceCudaStream);
  if (!status) {
    throw std::runtime_error("Inference execution failed");
  }
  
  // Collect output tensors
  rust::Vec<TensorInstance> outputs;
  
  for (int i = 0; i < static_cast<int>(mTensorMetadata.size()); ++i) {
    const auto &metadata = mTensorMetadata[i];
    
    if (metadata.ioMode != TensorIOMode::kOUTPUT) {
      continue; // skip if not tensor output
    }
    
    const Dims inferenceOutputShape = mContext->getTensorShape(metadata.name.c_str());
    
    TensorInstance output;
    size_t outputLen = metadata.dataTypeSize; // start with data type size
    for (int j = 0; j < inferenceOutputShape.nbDims; ++j) {
      outputLen *= inferenceOutputShape.d[j];
      output.shape.push_back(static_cast<int64_t>(inferenceOutputShape.d[j]));
    }
    
    // Create output tensor
    output.name = metadata.name;
    output.dtype = toTensorDataType(metadata.dataType);
    resize(output.data, outputLen);
    
    // Copy data from GPU buffer
    checkCudaErrorCode(cudaMemcpyAsync(output.data.data(), 
                                      static_cast<char*>(mBuffers[i]),
                                      outputLen, 
                                      cudaMemcpyDeviceToHost, mInferenceCudaStream));
    
    outputs.push_back(std::move(output));
  }
  
  checkCudaErrorCode(cudaStreamSynchronize(mInferenceCudaStream));
  
  return outputs;
}

rust::Vec<TensorInfo> Engine::get_input_tensor_info() const {
  rust::Vec<TensorInfo> result;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode == TensorIOMode::kINPUT) {
      TensorInfo info;
      info.name = metadata.name;
      resize(info.shape, 0);
      // Skip batch dimension (index 0)
      for (int j = 1; j < metadata.shape.nbDims; ++j) {
        info.shape.push_back(static_cast<uint32_t>(metadata.shape.d[j]));
      }
      info.dtype = toTensorDataType(metadata.dataType);
      result.push_back(std::move(info));
    }
  }
  return result;
}

rust::Vec<TensorInfo> Engine::get_output_tensor_info() const {
  rust::Vec<TensorInfo> result;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode == TensorIOMode::kOUTPUT) {
      TensorInfo info;
      info.name = metadata.name;
      resize(info.shape, 0);
      // Skip batch dimension (index 0)
      for (int j = 1; j < metadata.shape.nbDims; ++j) {
        info.shape.push_back(static_cast<uint32_t>(metadata.shape.d[j]));
      }
      info.dtype = toTensorDataType(metadata.dataType);
      result.push_back(std::move(info));
    }
  }
  return result;
}
