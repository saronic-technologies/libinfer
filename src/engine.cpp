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

namespace {
struct HostUnregisterPayload {
  std::vector<void*> ptrs;
};
}

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
    : mAsyncInferenceActive(false),
      mUsePinnedHostMemory(true),
      mValidateShapes(true),
      mEventH2DComplete(nullptr),
      mEventComputeComplete(nullptr),
      mCudaGraph(nullptr), mCudaGraphExec(nullptr),
      mGraphCaptured(false), mUseGraphOptimization(false),
      kEnginePath(options.path), kDeviceIndex(options.device_index) {
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

  // Cleanup CUDA Graph resources
  if (mCudaGraphExec != nullptr) {
    checkCudaErrorCode(cudaGraphExecDestroy(mCudaGraphExec));
  }
  if (mCudaGraph != nullptr) {
    checkCudaErrorCode(cudaGraphDestroy(mCudaGraph));
  }

  // Destroy persistent events
  if (mEventH2DComplete != nullptr) { checkCudaErrorCode(cudaEventDestroy(mEventH2DComplete)); }
  if (mEventComputeComplete != nullptr) { checkCudaErrorCode(cudaEventDestroy(mEventComputeComplete)); }

  // Destroy all streams
  checkCudaErrorCode(cudaStreamDestroy(mHostToDeviceStream));
  checkCudaErrorCode(cudaStreamDestroy(mComputeStream));
  checkCudaErrorCode(cudaStreamDestroy(mDeviceToHostStream));

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

  // Setup multi-stream architecture for optimal performance
  setupStreams();

  // Use the H2D stream for initial setup operations
  cudaStream_t& stream = mHostToDeviceStream;

  // Allocate GPU memory for input and output buffers
  mTensorMetadata.clear();
  mTensorMetadata.reserve(mEngine->getNbIOTensors());

  // ASSUMPTION: we always use optimization profile 0
  // set the optimization profile to 0 so we can query output shapes after setting input shapes
  mContext->setOptimizationProfileAsync(0, stream);
  
  // set metadata for input tensors first.
  // This has to be done first because we can't find the max input shape until we have specified the inputs
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorShape = mEngine->getTensorShape(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);

    if (tensorType == TensorIOMode::kOUTPUT) {
      continue; // skip output tensors for now
    }

    if (tensorType != TensorIOMode::kINPUT) {
      throw std::runtime_error("Error, IO Tensor is neither an input or output!");
    }
    
    // Store tensor metadata to avoid repeated TensorRT queries during inference
    TensorMetadata metadata;
    metadata.name = std::string(tensorName);
    metadata.ioMode = tensorType;
    metadata.dataType = tensorDataType;
    metadata.dataTypeSize = getDataTypeSize(toTensorDataType(tensorDataType));
    metadata.shape = tensorShape;
    metadata.bufferIndex = i;
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

    spdlog::debug("Tensor: {}, buffer index: {}, non-dynamic size (bytes): {}, max size: {}, ptr: {}",
                  metadata.name, metadata.bufferIndex, metadata.nonDynamicSize, inputLen, mBuffers[i]);

    // set the input shape to the max shape so we can query output shapes later
    mContext->setInputShape(metadata.name.c_str(), metadata.maxShape);
    
    mTensorMetadata.push_back(std::move(metadata));
  }

  // check that all input dimensions have been specified correctly
  int32_t ok = mContext->inferShapes(0, nullptr);
  if (ok != 0) {
    throw std::runtime_error("Error, not all input dimensions specified correctly.");
  }

  // collect output names first
  std::vector<std::string> outputNames;
  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);

    if (tensorType == TensorIOMode::kINPUT) {
      continue; // skip input tensors for now
    }

    if (tensorType != TensorIOMode::kOUTPUT) {
      throw std::runtime_error("Error, IO Tensor is neither an input or output!");
    }

    outputNames.push_back(std::string(tensorName));
  }

  for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
    const auto tensorName = mEngine->getIOTensorName(i);
    const auto tensorType = mEngine->getTensorIOMode(tensorName);
    const auto tensorShape = mEngine->getTensorShape(tensorName);
    const auto tensorDataType = mEngine->getTensorDataType(tensorName);

    if (tensorType == TensorIOMode::kINPUT) {
      continue; // skip output tensors for now
    }

    if (tensorType != TensorIOMode::kOUTPUT) {
      throw std::runtime_error("Error, IO Tensor is neither an input or output!");
    }
    
    // Store tensor metadata to avoid repeated TensorRT queries during inference
    TensorMetadata metadata;
    metadata.name = std::string(tensorName);
    metadata.ioMode = tensorType;
    metadata.dataType = tensorDataType;
    metadata.dataTypeSize = getDataTypeSize(toTensorDataType(tensorDataType));
    metadata.shape = tensorShape;
    metadata.bufferIndex = i;

    // these shapes may have -1 for dynamic dimensions we don't need to know these
    metadata.minShape = mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kMIN);
    metadata.optShape = mEngine->getProfileShape(tensorName, 0, OptProfileSelector::kOPT);

    // get actual max shape from context after setting input shapes
    metadata.maxShape = mEngine->getTensorShape(tensorName);

    size_t nonDynamicSize = 1;
    for (int j = 0; j < tensorShape.nbDims; ++j) {
      if (tensorShape.d[j] == -1) {
        nonDynamicSize *= 1; // Treat dynamic dimensions as size 1
      } else {
        nonDynamicSize *= tensorShape.d[j];
      }
    }
    metadata.nonDynamicSize = nonDynamicSize * metadata.dataTypeSize;

    // find max output size in bytes
    const int64_t inputLen = mContext->getMaxOutputSize(metadata.name.c_str());

    // Allocate as much memory for the largest supported batch size.
    checkCudaErrorCode(
      cudaMallocAsync(&mBuffers[i],
        inputLen,
        stream
      )
    );

    spdlog::debug("Output Tensor: {}, buffer index: {}, non-dynamic size (bytes): {}, max size: {}, ptr: {}",
                  metadata.name, metadata.bufferIndex, metadata.nonDynamicSize, inputLen, mBuffers[i]);
    
    mTensorMetadata.push_back(std::move(metadata));
  }

  // Now set metadata for output tensors

  // Synchronize the setup stream
  checkCudaErrorCode(cudaStreamSynchronize(stream));
}


rust::Vec<TensorInfo> Engine::get_input_tensor_info() const {
  rust::Vec<TensorInfo> result;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode == TensorIOMode::kINPUT) {
      TensorInfo info;
      info.name = metadata.name;
      resize(info.shape, 0);
      for (int j = 0; j < metadata.shape.nbDims; ++j) {
        info.shape.push_back(metadata.shape.d[j]);
        info.min_shape.push_back(metadata.minShape.d[j]);
        info.opt_shape.push_back(metadata.optShape.d[j]);
        info.max_shape.push_back(metadata.maxShape.d[j]);
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
      for (int j = 0; j < metadata.shape.nbDims; ++j) {
        info.shape.push_back(metadata.shape.d[j]);
        info.min_shape.push_back(metadata.minShape.d[j]);
        info.opt_shape.push_back(metadata.optShape.d[j]);
        info.max_shape.push_back(metadata.maxShape.d[j]);
      }
      info.dtype = toTensorDataType(metadata.dataType);
      result.push_back(std::move(info));
    }
  }
  return result;
}

// Setup multi-stream architecture for optimal performance
void Engine::setupStreams() {
  // Create separate streams for different operations to enable overlap
  checkCudaErrorCode(cudaStreamCreateWithFlags(&mHostToDeviceStream, cudaStreamNonBlocking));
  checkCudaErrorCode(cudaStreamCreateWithFlags(&mComputeStream, cudaStreamNonBlocking));
  checkCudaErrorCode(cudaStreamCreateWithFlags(&mDeviceToHostStream, cudaStreamNonBlocking));
  
  // Create reusable events (no timing) to minimize overhead
  checkCudaErrorCode(cudaEventCreateWithFlags(&mEventH2DComplete, cudaEventDisableTiming));
  checkCudaErrorCode(cudaEventCreateWithFlags(&mEventComputeComplete, cudaEventDisableTiming));
  
  spdlog::debug("Multi-stream architecture initialized: H2D={}, Compute={}, D2H={}", 
                (void*)mHostToDeviceStream, (void*)mComputeStream, (void*)mDeviceToHostStream);
}

// Capture CUDA graph for repeated inference patterns
void Engine::captureGraph() {
  if (mGraphCaptured) {
    return;
  }
  
  spdlog::info("Capturing CUDA graph for optimized inference");
  
  // Start graph capture on the compute stream
  checkCudaErrorCode(cudaStreamBeginCapture(mComputeStream, cudaStreamCaptureModeGlobal));
  
  // Execute one inference to capture the pattern
  // Note: This assumes input shapes have been set appropriately
  bool status = mContext->enqueueV3(mComputeStream);
  if (!status) {
    throw std::runtime_error("Failed to capture CUDA graph - enqueueV3 failed");
  }
  
  // End capture and create executable graph
  checkCudaErrorCode(cudaStreamEndCapture(mComputeStream, &mCudaGraph));
  checkCudaErrorCode(cudaGraphInstantiate(&mCudaGraphExec, mCudaGraph, nullptr, nullptr, 0));
  
  mGraphCaptured = true;
  spdlog::info("CUDA graph capture completed successfully");
}

// Synchronize all streams
void Engine::synchronizeStreams() {
  checkCudaErrorCode(cudaStreamSynchronize(mHostToDeviceStream));
  checkCudaErrorCode(cudaStreamSynchronize(mComputeStream));
  checkCudaErrorCode(cudaStreamSynchronize(mDeviceToHostStream));
}

// Optimized synchronous inference with multi-stream architecture
rust::Vec<TensorInstance> Engine::infer(const rust::Vec<TensorInstance> &input) {
  // Start async inference
  infer_async(input);
  // Wait for completion and return results
  return wait_for_completion();
}

// Asynchronous inference implementation with multi-stream optimization
void Engine::infer_async(const rust::Vec<TensorInstance> &input) {
  if (mAsyncInferenceActive) {
    throw std::runtime_error("Cannot start new async inference while previous one is active. Call wait_for_completion() first.");
  }
  
  mAsyncInferenceActive = true;
  
  // Create a map from tensor name to input data for easy lookup
  std::unordered_map<std::string, const TensorInstance*> inputMap;
  for (const auto &tensorInput : input) {
    inputMap[std::string(tensorInput.name)] = &tensorInput;
  }
  
  HostUnregisterPayload* unregisterPayload = nullptr;
  if (mUsePinnedHostMemory) { unregisterPayload = new HostUnregisterPayload(); }
  
  // Phase 1: Input validation and shape setting (CPU operations)
  for (int i = 0; i < static_cast<int>(mTensorMetadata.size()); ++i) {
    const auto &metadata = mTensorMetadata[i];
    
    if (metadata.ioMode != TensorIOMode::kINPUT) {
      continue; // If not a tensor input skip
    }
    
    // Find the corresponding input data
    auto it = inputMap.find(metadata.name);
    if (it == inputMap.end()) {
      mAsyncInferenceActive = false;
      throw std::runtime_error("Missing input tensor: " + metadata.name);
    }
    
    const auto &tensorInput = *it->second;

    // validate input shape is valid with input size 
    if (static_cast<int32_t>(tensorInput.shape.size()) != metadata.shape.nbDims) {
      mAsyncInferenceActive = false;
      throw std::runtime_error("Input tensor '" + metadata.name + 
                              "' has incorrect number of dimensions: " + 
                              std::to_string(tensorInput.shape.size()) + 
                              ", expected: " + std::to_string(metadata.shape.nbDims));
    }

    size_t inputShapeSize = 1;
    for (size_t j = 0; j < tensorInput.shape.size(); ++j) {
      if (tensorInput.shape[j] < 1) {
        mAsyncInferenceActive = false;
        throw std::runtime_error(
                                "Invalid dimension size in input tensor '" + metadata.name + "'" + 
                                " at index " + std::to_string(j) + ": " + std::to_string(tensorInput.shape[j])
        );
      }

      if (tensorInput.shape[j] < metadata.minShape.d[j]) {
        mAsyncInferenceActive = false;
        throw std::runtime_error("Input tensor '" + metadata.name + 
                                "' dimension " + std::to_string(j) + 
                                " is smaller than minimum: " + 
                                std::to_string(tensorInput.shape[j]) + 
                                " < " + std::to_string(metadata.minShape.d[j]));
      }

      if (tensorInput.shape[j] > metadata.maxShape.d[j]) {
        mAsyncInferenceActive = false;
        throw std::runtime_error("Input tensor '" + metadata.name + 
                                "' dimension " + std::to_string(j) + 
                                " is larger than maximum: " + 
                                std::to_string(tensorInput.shape[j]) + 
                                " > " + std::to_string(metadata.maxShape.d[j]));
      }
      
      inputShapeSize *= tensorInput.shape[j];
    }

    inputShapeSize *= metadata.dataTypeSize;
    if (inputShapeSize != tensorInput.data.size()) {
      mAsyncInferenceActive = false;
      throw std::runtime_error("Input tensor '" + metadata.name + 
                              "' has size mismatch calculated bytes from input tensor shape: " + 
                              std::to_string(inputShapeSize) + 
                              " != " + std::to_string(tensorInput.data.size()));
    }
    
    // set input shape from tensor input shape
    nvinfer1::Dims inputDims = nvinfer1::Dims{};
    inputDims.nbDims = static_cast<int32_t>(tensorInput.shape.size());
    for (size_t j = 0; j < tensorInput.shape.size(); ++j) {
      inputDims.d[j] = tensorInput.shape[j];
    }

    bool shapeStatus = mContext->setInputShape(metadata.name.c_str(), inputDims);
    if (!shapeStatus) {
      mAsyncInferenceActive = false;
      throw std::runtime_error("Failed to set input shape for tensor: " + metadata.name);
    }
  }
  
  // Phase 2: Async host-to-device memory transfers
  for (int i = 0; i < static_cast<int>(mTensorMetadata.size()); ++i) {
    const auto &metadata = mTensorMetadata[i];
    
    if (metadata.ioMode != TensorIOMode::kINPUT) {
      continue;
    }
    
    auto it = inputMap.find(metadata.name);
    const auto &tensorInput = *it->second;
    
    // Copy input data to GPU buffer using dedicated H2D stream
    // Optionally register host memory as pinned for faster transfers
    if (mUsePinnedHostMemory) {
      cudaError_t reg = cudaHostRegister((void*)tensorInput.data.data(), tensorInput.data.size(), 0);
      if (reg == cudaSuccess) { unregisterPayload->ptrs.push_back((void*)tensorInput.data.data()); }
      else { spdlog::warn("cudaHostRegister failed for input '{}' ({}), continuing without pinned host memory", metadata.name, (int)reg); }
    }
    checkCudaErrorCode(cudaMemcpyAsync(mBuffers[metadata.bufferIndex], tensorInput.data.data(), 
                                      tensorInput.data.size(),
                                      cudaMemcpyHostToDevice, mHostToDeviceStream));
  }
  
  // Ensure all dynamic bindings have been defined
  if (!mContext->allInputDimensionsSpecified()) {
    mAsyncInferenceActive = false;
    throw std::runtime_error("Error, not all required dimensions specified.");
  }
  
  // Set the address of all input and output buffers using cached names
  for (int i = 0; i < static_cast<int>(mTensorMetadata.size()); ++i) {
    spdlog::debug("Setting tensor address for: {}, buffer index: {}, buffer ptr: {}",
                  mTensorMetadata[i].name, mTensorMetadata[i].bufferIndex, mBuffers[mTensorMetadata[i].bufferIndex]);

    bool status = mContext->setTensorAddress(mTensorMetadata[i].name.c_str(), mBuffers[mTensorMetadata[i].bufferIndex]);
    if (!status) {
      mAsyncInferenceActive = false;
      throw std::runtime_error("Unable to set tensor address for: " + mTensorMetadata[i].name);
    }
  }

  // Get output tensor names from metadata
  std::vector<const char*> outputNames;
  for (const auto &metadata : mTensorMetadata) {
    if (metadata.ioMode == TensorIOMode::kOUTPUT) {
      outputNames.push_back(metadata.name.c_str());
    }
  }

  // verify output shapes (optional)
  if (mValidateShapes) {
  const int32_t error_code = mContext->inferShapes(static_cast<int32_t>(outputNames.size()), outputNames.data());
  if (error_code > 0) {
    mAsyncInferenceActive = false;
    std::string namesCombined;
    for (size_t i = 0; i < outputNames.size(); ++i) {
      namesCombined += outputNames[i];
      if (i < outputNames.size() - 1) {
        namesCombined += ", ";
      }
    }
    throw std::runtime_error(
      "Failed to infer output shapes, for output names : " + namesCombined
      + ", " + std::to_string(error_code) + " input tensors not specified correctly.");
  }
  
  }
  // Phase 3: Use persistent CUDA events for proper stream synchronization
  // Record completion of H2D transfers
  checkCudaErrorCode(cudaEventRecord(mEventH2DComplete, mHostToDeviceStream));
  // If pinned host memory used for inputs, unregister after H2D completes (host callback)
  if (mUsePinnedHostMemory && unregisterPayload) {
    checkCudaErrorCode(cudaLaunchHostFunc(mHostToDeviceStream, [](void* data){
      auto payload = static_cast<HostUnregisterPayload*>(data);
      for (void* p : payload->ptrs) { cudaHostUnregister(p); }
      delete payload;
    }, unregisterPayload));
  }
  // Wait for H2D transfers to complete before starting inference
  checkCudaErrorCode(cudaStreamWaitEvent(mComputeStream, mEventH2DComplete, 0));
  
  // Phase 4: Run inference - use CUDA Graph if available, otherwise standard execution
  if (mUseGraphOptimization && mGraphCaptured) {
    checkCudaErrorCode(cudaGraphLaunch(mCudaGraphExec, mComputeStream));
  } else {
    bool status = mContext->enqueueV3(mComputeStream);
    if (!status) {
      mAsyncInferenceActive = false;
      throw std::runtime_error("Inference execution failed");
    }
    
    // Attempt to capture CUDA graph after first successful inference
    if (mUseGraphOptimization && !mGraphCaptured) {
      try {
        captureGraph();
      } catch (const std::exception& e) {
        spdlog::warn("CUDA graph capture failed: {}", e.what());
        mUseGraphOptimization = false;
      }
    }
  }
  
  // Record completion of compute
  checkCudaErrorCode(cudaEventRecord(mEventComputeComplete, mComputeStream));
  
  // Wait for inference to complete before starting D2H transfers
  checkCudaErrorCode(cudaStreamWaitEvent(mDeviceToHostStream, mEventComputeComplete, 0));
  
  // Phase 5: Start async device-to-host transfers
  mPendingOutputs.clear();
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
    
    // Optionally register output host buffer for faster D2H
    if (mUsePinnedHostMemory) {
      cudaError_t reg = cudaHostRegister((void*)output.data.data(), outputLen, 0);
      if (reg == cudaSuccess) { mRegisteredOutputHostPtrs.push_back((void*)output.data.data()); }
    }
    
    // Start D2H transfer
    checkCudaErrorCode(cudaMemcpyAsync(output.data.data(), 
                                      static_cast<char*>(mBuffers[metadata.bufferIndex]),
                                      outputLen, 
                                      cudaMemcpyDeviceToHost, mDeviceToHostStream));
    
    mPendingOutputs.push_back(std::move(output));
  }
  
}

// Wait for async inference completion
rust::Vec<TensorInstance> Engine::wait_for_completion() {
  if (!mAsyncInferenceActive) {
    throw std::runtime_error("No active async inference to wait for. Call infer_async() first.");
  }
  
  // Wait for all D2H transfers to complete
  checkCudaErrorCode(cudaStreamSynchronize(mDeviceToHostStream));
  
  if (mUsePinnedHostMemory) {
    for (void* p : mRegisteredOutputHostPtrs) { cudaHostUnregister(p); }
    mRegisteredOutputHostPtrs.clear();
  }
  
  mAsyncInferenceActive = false;
  
  // Move pending outputs to return vector
  rust::Vec<TensorInstance> outputs;
  for (auto& output : mPendingOutputs) {
    outputs.push_back(std::move(output));
  }
  mPendingOutputs.clear();
  
  return outputs;
}

// Enable CUDA Graph optimization
void Engine::enable_cuda_graphs() {
  if (mGraphCaptured) {
    spdlog::warn("CUDA Graph already captured and enabled");
    return;
  }
  
  mUseGraphOptimization = true;
  // Graph will be captured on next inference
  spdlog::info("CUDA Graph optimization enabled. Graph will be captured on next inference.");
}

// Check if async inference is complete
bool Engine::is_inference_complete() const {
  if (!mAsyncInferenceActive) {
    return true;
  }
  
  // Check if D2H stream is complete (non-blocking)
  cudaError_t result = cudaStreamQuery(mDeviceToHostStream);
  return result == cudaSuccess;
}
void Engine::enable_pinned_memory(bool enable) {
  mUsePinnedHostMemory = enable;
  spdlog::info("Pinned host memory {}", enable ? "ENABLED" : "DISABLED");
}

void Engine::set_validation_enabled(bool enable) {
  mValidateShapes = enable;
  spdlog::info("Runtime validation {}", enable ? "ENABLED" : "DISABLED");
}
