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
std::unique_ptr<Engine> make_engine(const Options &options) {
  auto engine = std::make_unique<Engine>(options);
  engine->build();
  engine->load();
  return engine;
}

// A few utility functions
static inline void checkCudaErrorCode(cudaError_t code) {
  if (code != 0) {
    std::string errMsg =
        "CUDA operation failed with code: " + std::to_string(code) + "(" +
        cudaGetErrorName(code) + "), with message: " + cudaGetErrorString(code);
    throw std::runtime_error(errMsg);
  }
}

static inline bool doesFileExist(const std::string &filepath) {
  std::ifstream f(filepath.c_str());
  return f.good();
}

// Pretty dumb that we don't get a resize method in rust::Vec.
template <typename T> static void resize(rust::Vec<T> &v, size_t len) {
  v.reserve(len);
  while (v.size() < len) {
    v.push_back(T{});
  }
}

Engine::Engine(const Options &options)
    : kModelName(options.model_name), kSearchPath(options.search_path),
      kSavePath(options.save_path),
      kCanonicalEngineName(serializeEngineOptions(options)),
      kPrecision(static_cast<uint8_t>(options.precision)),
      kDeviceIndex(options.device_index),
      kOptBatchSize(options.optimized_batch_size),
      kMaxBatchSize(options.max_batch_size) {
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

void Engine::build() {
  const auto primaryPath = std::filesystem::path(kSearchPath) /
                           std::filesystem::path(kCanonicalEngineName);
  const auto secondaryPath = std::filesystem::path(kSavePath) /
                             std::filesystem::path(kCanonicalEngineName);

  // Check if engine exists at search path.
  if (doesFileExist(primaryPath)) {
    spdlog::info("Found engine {}", primaryPath.string());
    mEnginePath = primaryPath;
    return;
  } else {
    mEnginePath = secondaryPath;
  }

  // Check if engine exists at save path.
  if (doesFileExist(mEnginePath)) {
    spdlog::info("Found engine {}", secondaryPath.string());
    return;
  }

  const auto onnxPath = std::filesystem::path(kSearchPath) /
                        std::filesystem::path(kModelName + ".onnx");

  if (!doesFileExist(onnxPath)) {
    throw std::runtime_error("Could not find onnx model at path: " +
                             onnxPath.string());
  }

  spdlog::info("Engine not found, generating. This could take a while...");

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(mLogger));
  if (!builder) {
    throw std::runtime_error("Could not create engine builder");
  }

  // Define an explicit batch size and then create the network (implicit batch
  // size is deprecated). More info here:
  // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
  auto explicitBatch = 1U << static_cast<uint32_t>(
                           NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatch));
  if (!network) {
    throw std::runtime_error("Could not create network definition");
  }

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, mLogger));
  if (!parser) {
    throw std::runtime_error("Could nto create onnx parser");
  }

  // We are going to first read the onnx file into memory, then pass that buffer
  // to the parser. Had our onnx model file been encrypted, this approach would
  // allow us to first decrypt the buffer.
  std::ifstream file(onnxPath, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Unable to read onnx file");
  }

  // Parse the buffer we read into memory.
  auto parsed = parser->parse(buffer.data(), buffer.size());
  if (!parsed) {
    throw std::runtime_error("Could not parse onnx file");
  }

  // Ensure that all the inputs have the same batch size
  const auto numInputs = network->getNbInputs();
  if (numInputs < 1) {
    throw std::runtime_error("Error, model needs at least 1 input!");
  }
  const auto input0Batch = network->getInput(0)->getDimensions().d[0];
  for (int32_t i = 1; i < numInputs; ++i) {
    if (network->getInput(i)->getDimensions().d[0] != input0Batch) {
      throw std::runtime_error("Error, the model has multiple inputs, each "
                               "with differing batch sizes!");
    }
  }

  // Check to see if the model supports dynamic batch size or not
  bool doesSupportDynamicBatch = false;
  if (input0Batch == -1) {
    doesSupportDynamicBatch = true;
    spdlog::info("Model supports dynamic batch size");
  } else {
    spdlog::info("Model requires fixed batch size of {}", input0Batch);
    // If the model supports a fixed batch size, ensure that the maxBatchSize
    // and optBatchSize were set correctly.
    if (kOptBatchSize != input0Batch || kMaxBatchSize != input0Batch) {
      throw std::runtime_error(
          "Error, model requires a fixed batch size of " +
          std::to_string(input0Batch) +
          ". Must set Options.optimized_batch_size and Options.max_batch_size "
          "to this value.");
    }
  }

  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    throw std::runtime_error("Could not create builder config");
  }

  // Register a single optimization profile
  IOptimizationProfile *optProfile = builder->createOptimizationProfile();
  for (int32_t i = 0; i < numInputs; ++i) {
    // Must specify dimensions for all the inputs the model expects.
    const auto input = network->getInput(i);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    int32_t inputC = inputDims.d[1];
    int32_t inputH = inputDims.d[2];
    int32_t inputW = inputDims.d[3];

    // Specify the optimization profile.
    if (doesSupportDynamicBatch) {
      optProfile->setDimensions(inputName, OptProfileSelector::kMIN,
                                Dims4(1, inputC, inputH, inputW));
    } else {
      optProfile->setDimensions(inputName, OptProfileSelector::kMIN,
                                Dims4(kOptBatchSize, inputC, inputH, inputW));
    }
    optProfile->setDimensions(inputName, OptProfileSelector::kOPT,
                              Dims4(kOptBatchSize, inputC, inputH, inputW));
    optProfile->setDimensions(inputName, OptProfileSelector::kMAX,
                              Dims4(kMaxBatchSize, inputC, inputH, inputW));
  }
  config->addOptimizationProfile(optProfile);

  if (static_cast<Precision>(kPrecision) == Precision::FP16) {
    if (!builder->platformHasFastFp16()) {
      throw std::runtime_error("Error: GPU does not support FP16 precision");
    }
    config->setFlag(BuilderFlag::kFP16);
  } else if (static_cast<Precision>(kPrecision) == Precision::INT8) {
    throw std::runtime_error("INT8 is not yet supported");
  }

  // CUDA stream used for profiling by the builder.
  cudaStream_t profileStream;
  checkCudaErrorCode(cudaStreamCreate(&profileStream));
  config->setProfileStream(profileStream);

  // Build the engine.
  // If this call fails, it is suggested to increase the logger verbosity to
  // kVERBOSE and try rebuilding the engine. Doing so will provide you with more
  // information on why exactly it is failing.
  std::unique_ptr<IHostMemory> plan{
      builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    throw std::runtime_error("Could not create network serializer");
  }

  // Write the engine to disk (create output folder if necessary)
  if (std::filesystem::create_directories(std::filesystem::path(kSavePath))) {
    spdlog::info("Created output folder for engines at {}", kSavePath);
  }
  std::ofstream outfile(mEnginePath, std::ofstream::binary);
  if (!outfile.is_open()) {
    throw std::runtime_error("Could not open output file");
  }

  outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
  if (outfile.fail()) {
    throw std::runtime_error("Could not write engine file to disk");
  }

  outfile.close();
  if (outfile.fail()) {
    throw std::runtime_error("Could not close engine file");
  }

  spdlog::info("Saved engine to {}", mEnginePath);

  checkCudaErrorCode(cudaStreamDestroy(profileStream));
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
  std::ifstream file(mEnginePath, std::ios::binary | std::ios::ate);
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
    if (tensorType == TensorIOMode::kINPUT) {
      checkCudaErrorCode(
          cudaMallocAsync(&mBuffers[i],
                          kMaxBatchSize * tensorShape.d[1] * tensorShape.d[2] *
                              tensorShape.d[3] * sizeof(float),
                          stream));

      // Store the input dims for later use
      mInputDims.emplace_back(tensorShape.d[1], tensorShape.d[2],
                              tensorShape.d[3]);
      mInputBatchSize = tensorShape.d[0];
    } else if (tensorType == TensorIOMode::kOUTPUT) {
      // The binding is an output
      uint32_t outputLenFloat = 1;
      mOutputDims.push_back(tensorShape);

      for (int j = 1; j < tensorShape.nbDims; ++j) {
        // We ignore j = 0 because that is the batch size, and we will take that
        // into account when sizing the buffer
        outputLenFloat *= tensorShape.d[j];
      }

      mOutputLengths.push_back(outputLenFloat);
      checkCudaErrorCode(cudaMallocAsync(
          &mBuffers[i], outputLenFloat * kMaxBatchSize * sizeof(float),
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

rust::Vec<float> Engine::infer(const rust::Vec<float> &input) {
  // Create the cuda stream that will be used for inference
  cudaStream_t inferenceCudaStream;
  checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

  const auto &dims = mInputDims[0];

  // Check that the passed batch size can be handled.
  const int32_t calculatedBatchSize =
      input.size() / (dims.d[0] * dims.d[1] * dims.d[2]);
  if (calculatedBatchSize > kMaxBatchSize) {
    throw std::runtime_error(
        "Input exceeds max batch size: " + std::to_string(calculatedBatchSize) +
        " > " + std::to_string(kMaxBatchSize));
  }

  // If the network's batch size is fixed, the input batch dimension must match.
  if (mInputBatchSize != -1 && calculatedBatchSize != mInputBatchSize) {
    throw std::runtime_error(
        "Input batch size does not match required fixed batch size: " +
        std::to_string(calculatedBatchSize) +
        " != " + std::to_string(kOptBatchSize));
  }

  // Check that vector has enough elements for full input.
  if (input.size() % (dims.d[0] * dims.d[1] * dims.d[2]) != 0) {
    throw std::runtime_error("Input vector incorrectly sized");
  }

  // Define the batch size.
  nvinfer1::Dims4 inputDims = {calculatedBatchSize, dims.d[0], dims.d[1],
                               dims.d[2]};
  mContext->setInputShape(mIOTensorNames[0].c_str(), inputDims);

  checkCudaErrorCode(
      cudaMemcpyAsync(mBuffers[0], input.data(), input.size() * sizeof(float),
                      cudaMemcpyHostToDevice, inferenceCudaStream));

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

std::string Engine::serializeEngineOptions(const Options &options) {
  auto canonicalName = std::string(options.model_name);

  // Add the GPU device name to the file to ensure that the model is only used
  // on devices with the exact same GPU.
  std::vector<std::string> deviceNames;
  getDeviceNames(deviceNames);

  if (static_cast<size_t>(options.device_index) >= deviceNames.size()) {
    throw std::runtime_error("Provided device index is out of range");
  }

  auto deviceName = deviceNames[options.device_index];
  deviceName.erase(
      std::remove_if(deviceName.begin(), deviceName.end(), ::isspace),
      deviceName.end());

  canonicalName += "_" + deviceName;

  // Serialize the specified options into the filename.
  if (options.precision == Precision::FP16) {
    canonicalName += "_fp16";
  } else if (options.precision == Precision::FP32) {
    canonicalName += "_fp32";
  } else {
    canonicalName += "_int8";
  }

  canonicalName += "_b" + std::to_string(options.optimized_batch_size) + "m" +
                   std::to_string(options.max_batch_size);

  canonicalName += ".engine";

  return canonicalName;
}

void Engine::getDeviceNames(std::vector<std::string> &deviceNames) {
  int numGPUs;
  cudaGetDeviceCount(&numGPUs);

  for (int device = 0; device < numGPUs; device++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    deviceNames.push_back(std::string(prop.name));
  }
}
