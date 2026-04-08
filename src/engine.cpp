#include "engine.h"

#include <NvInferRuntimeBase.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sched.h>
#include <set>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unordered_map>

#include "libinfer/src/lib.rs.h"

#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>


using namespace nvinfer1;

// Enumerate all thread IDs in the current process via /proc/self/task/.
static std::set<pid_t> enumerate_threads() {
  std::set<pid_t> tids;
  DIR *dir = opendir("/proc/self/task");
  if (!dir)
    return tids;
  while (struct dirent *entry = readdir(dir)) {
    if (entry->d_name[0] == '.')
      continue;
    tids.insert(static_cast<pid_t>(std::atoi(entry->d_name)));
  }
  closedir(dir);
  return tids;
}

// Format a cpu_set_t as a human-readable list like "0-3,8-11".
static std::string format_cpuset(const cpu_set_t &cpuset) {
  std::string result;
  int count = CPU_COUNT(&cpuset);
  int ncpus = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
  bool in_range = false;
  int range_start = -1;

  auto flush_range = [&](int end) {
    if (!result.empty())
      result += ",";
    if (range_start == end)
      result += std::to_string(range_start);
    else
      result += std::to_string(range_start) + "-" + std::to_string(end);
  };

  for (int i = 0; i < ncpus; ++i) {
    if (CPU_ISSET(i, &cpuset)) {
      if (!in_range) {
        range_start = i;
        in_range = true;
      }
    } else if (in_range) {
      flush_range(i - 1);
      in_range = false;
    }
  }
  if (in_range)
    flush_range(ncpus - 1);

  return result + " (" + std::to_string(count) + " cpus)";
}

// Log thread name and affinity for a given TID.
static void log_thread_info(pid_t tid, const std::string &label) {
  // Read thread name from /proc/self/task/<tid>/comm
  std::string name = "<unknown>";
  std::string comm_path = "/proc/self/task/" + std::to_string(tid) + "/comm";
  std::ifstream comm_file(comm_path);
  if (comm_file.is_open()) {
    std::getline(comm_file, name);
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  sched_getaffinity(tid, sizeof(cpuset), &cpuset);

  spdlog::info("  [{}] tid={} name={} affinity={}", label, tid, name,
               format_cpuset(cpuset));
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
          mInferenceCudaStream
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
          mInferenceCudaStream));
    } else {
      throw std::runtime_error(
          "Error, IO Tensor is neither an input or output!");
    }
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
  
  // Run inference.
  bool status = mContext->enqueueV3(mInferenceCudaStream);
  if (!status) {
    throw std::runtime_error("Inference execution failed");
  }

  // TRT creates hardware_concurrency-1 worker threads that spin-wait with
  // sched_yield. These threads have consecutive TIDs and inherit the name of
  // the thread that triggered their creation. We detect them by finding the
  // longest run of consecutive TIDs sharing the same name, then lower their
  // scheduling priority so they don't starve application threads.
  if (mFirstInfer) {
    mFirstInfer = false;

    static std::once_flag census_flag;
    std::call_once(census_flag, [this]() {
      auto all_tids = enumerate_threads();
      int ncpus = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
      int expected_workers = ncpus - 1;

      spdlog::info("=== TRT worker detection ({} threads, expecting ~{} "
                   "workers) ===",
                   all_tids.size(), expected_workers);

      // Build sorted vector of (tid, name) pairs
      struct ThreadInfo {
        pid_t tid;
        std::string name;
      };
      std::vector<ThreadInfo> threads;
      for (pid_t tid : all_tids) {
        std::string name;
        std::string path =
            "/proc/self/task/" + std::to_string(tid) + "/comm";
        std::ifstream f(path);
        if (f.is_open()) {
          std::getline(f, name);
        }
        threads.push_back({tid, name});
      }

      // Sort by TID (enumerate_threads returns a set, already sorted)
      std::sort(threads.begin(), threads.end(),
                [](const ThreadInfo &a, const ThreadInfo &b) {
                  return a.tid < b.tid;
                });

      // Find the longest run of consecutive TIDs sharing the same name,
      // excluding wrapper threads (name contains "wrapp" from the
      // truncated process name e.g. ".torchyd2-wrapp").
      size_t best_start = 0, best_len = 0;
      size_t run_start = 0;
      for (size_t i = 1; i < threads.size(); ++i) {
        bool consecutive = (threads[i].tid == threads[i - 1].tid + 1);
        bool same_name = (threads[i].name == threads[run_start].name);
        bool is_wrapper =
            threads[i].name.find("wrapp") != std::string::npos;
        if (consecutive && same_name && !is_wrapper) {
          size_t run_len = i - run_start + 1;
          if (run_len > best_len) {
            best_start = run_start;
            best_len = run_len;
          }
        } else {
          run_start = i;
        }
      }

      spdlog::info("  Longest consecutive-TID run: {} threads (TID {}-{}), "
                   "name=\"{}\"",
                   best_len,
                   best_len > 0 ? threads[best_start].tid : 0,
                   best_len > 0 ? threads[best_start + best_len - 1].tid : 0,
                   best_len > 0 ? threads[best_start].name : "");

      // Only act if the run is close to expected_workers (within ±4)
      bool is_trt_pool =
          best_len >= static_cast<size_t>(std::max(1, expected_workers - 4));

      if (is_trt_pool) {
        std::vector<pid_t> trt_tids;
        for (size_t i = best_start; i < best_start + best_len; ++i) {
          trt_tids.push_back(threads[i].tid);
        }

        // Pin TRT workers to the last 5 CPUs, freeing the rest for
        // application threads (preprocessing, tracking, depth, etc.).
        constexpr int kTrtCpus = 6;
        int pin_start = std::max(0, ncpus - kTrtCpus);
        cpu_set_t trt_cpuset;
        CPU_ZERO(&trt_cpuset);
        for (int c = pin_start; c < ncpus; ++c) {
          CPU_SET(c, &trt_cpuset);
        }

        spdlog::info("  Identified {} TRT worker threads, pinning to CPUs "
                     "{}-{}",
                     trt_tids.size(), pin_start, ncpus - 1);

        for (pid_t tid : trt_tids) {
          int ret = sched_setaffinity(tid, sizeof(trt_cpuset), &trt_cpuset);
          if (ret != 0) {
            spdlog::warn("  sched_setaffinity(tid={}) failed: {}", tid,
                         strerror(errno));
          }

          log_thread_info(tid, ret == 0 ? "pinned" : "FAILED");
        }
      } else {
        spdlog::info("  No TRT worker pool detected (longest run {} < "
                     "expected {})",
                     best_len, expected_workers);
        // Log all threads for debugging
        for (const auto &t : threads) {
          log_thread_info(t.tid, t.name);
        }
      }
    });
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
    
    // Create output tensor with bulk-allocated buffer.
    OutputTensor output;
    size_t copySize = outputLen * metadata.dataTypeSize;
    output.name = metadata.name;
    output.dtype = toTensorDataType(metadata.dataType);
    output.data = new_output_buffer(copySize);
    
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
