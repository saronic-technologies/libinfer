//! Device Tensor Comparison Example
//!
//! Compares performance of CPU vs device-resident tensors for TensorRT inference

use anyhow::{anyhow, Result};
use clap::Parser;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use libinfer::{DeviceTensor, Engine, Options, TensorDataType, TensorInstance};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn, Level};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
#[clap(about = "Compare CPU vs device tensor performance for TensorRT")]
struct Args {
    /// Path to the TensorRT engine file
    #[arg(value_name = "ENGINE_PATH")]
    engine: PathBuf,
}

/// Generate test data based on tensor info and random seed
fn generate_test_data_seeded(tensor_info: &libinfer::TensorInfo, rng: &mut StdRng) -> Vec<u8> {
    let shape: Vec<i64> = tensor_info
        .shape
        .iter()
        .map(|&d| if d == -1 { 1 } else { d })
        .collect();

    let num_elements: usize = shape.iter().map(|&d| d as usize).product();

    match tensor_info.dtype {
        TensorDataType::FP32 => {
            let data: Vec<f32> = (0..num_elements)
                .map(|_| rng.gen_range(-1.0..=1.0))
                .collect();
            data.iter().flat_map(|&f| f.to_le_bytes()).collect()
        }
        TensorDataType::UINT8 => (0..num_elements).map(|_| rng.gen_range(0..=255)).collect(),
        TensorDataType::INT64 => {
            let data: Vec<i64> = (0..num_elements)
                .map(|_| rng.gen_range(-1000..=1000))
                .collect();
            data.iter().flat_map(|&i| i.to_le_bytes()).collect()
        }
        TensorDataType::BOOL => (0..num_elements)
            .map(|_| if rng.gen_bool(0.5) { 1u8 } else { 0u8 })
            .collect(),
        _ => {
            panic!(
                "Unsupported tensor data type for random generation: {:?}",
                tensor_info.dtype
            )
        }
    }
}

fn main() -> Result<()> {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();

    // Set reasonable defaults
    let device_index = 0;
    let warmup = 200;  // Increased warmup to handle CUDA caching effects
    let benchmark_iterations = 1000;
    let correctness_iterations = 10;
    let seed = 42;
    let tolerance = 1e-4_f32;

    // Initialize CUDA device (safe)
    let device = CudaDevice::new(device_index as usize)?;
    info!("CUDA device {} initialized", device_index);

    // Initialize seeded RNG for reproducible results
    let mut rng = StdRng::seed_from_u64(seed);
    info!("Using random seed: {}", seed);

    // Load TensorRT engine
    info!("Loading TensorRT engine: {}", args.engine.display());
    let options = Options {
        path: args.engine.to_string_lossy().to_string(),
        device_index,
    };

    let mut engine = Engine::new(&options)?;
    info!("Engine loaded successfully");

    // Get tensor information
    let input_infos = engine.get_input_tensor_info();
    let output_infos = engine.get_output_tensor_info();

    if input_infos.is_empty() {
        return Err(anyhow!("No input tensors found"));
    }

    info!(
        "Model has {} inputs and {} outputs",
        input_infos.len(),
        output_infos.len()
    );

    // Print detailed tensor information
    for (i, input_info) in input_infos.iter().enumerate() {
        let tensor_size: usize = input_info
            .shape
            .iter()
            .map(|&d| if d == -1 { 1 } else { d as usize })
            .product();
        info!(
            "Input {}: '{}' shape={:?} dtype={:?} size={}",
            i, input_info.name, input_info.shape, input_info.dtype, tensor_size
        );
    }

    for (i, output_info) in output_infos.iter().enumerate() {
        let tensor_size: usize = output_info
            .shape
            .iter()
            .map(|&d| if d == -1 { 1 } else { d as usize })
            .product();
        info!(
            "Output {}: '{}' shape={:?} dtype={:?} size={}",
            i, output_info.name, output_info.shape, output_info.dtype, tensor_size
        );
    }

    // Generate test data for benchmarking and comparison
    info!("Generating test data with random seed {}...", seed);
    let mut cpu_inputs = Vec::new();
    for input_info in &input_infos {
        let shape: Vec<i64> = input_info
            .shape
            .iter()
            .map(|&d| if d == -1 { 1 } else { d })
            .collect();

        let data = generate_test_data_seeded(input_info, &mut rng);

        info!(
            "Input '{}': shape={:?}, size={} bytes",
            input_info.name,
            shape,
            data.len()
        );

        cpu_inputs.push(TensorInstance {
            name: input_info.name.clone(),
            data,
            shape,
            dtype: input_info.dtype.clone(),
        });
    }

    // Prepare device tensors (safe memory management)
    let mut device_inputs = Vec::new();
    let mut device_outputs = Vec::new();
    let mut device_slices: Vec<CudaSlice<u8>> = Vec::new(); // Safe CUDA memory

    // Allocate and copy input tensors to device (safe)
    for cpu_input in &cpu_inputs {
        let device_slice = device.htod_copy(cpu_input.data.clone())?;
        let device_ptr = *device_slice.device_ptr() as *mut u8;

        device_inputs.push(DeviceTensor {
            name: cpu_input.name.clone(),
            device_ptr,
            size_bytes: cpu_input.data.len(),
            shape: cpu_input.shape.clone(),
            dtype: cpu_input.dtype.clone(),
        });

        device_slices.push(device_slice);
    }

    // Allocate output tensors on device (safe)
    for output_info in &output_infos {
        let shape: Vec<i64> = output_info
            .shape
            .iter()
            .map(|&d| if d == -1 { 1 } else { d })
            .collect();

        let num_elements: usize = shape.iter().map(|&d| d as usize).product();
        let size_bytes = match &output_info.dtype {
            &TensorDataType::FP32 => num_elements * 4,
            &TensorDataType::UINT8 => num_elements,
            &TensorDataType::INT64 => num_elements * 8,
            &TensorDataType::BOOL => num_elements,
            _ => panic!("Unsupported output tensor data type"),
        };

        let device_slice = device.alloc_zeros::<u8>(size_bytes)?;
        let device_ptr = *device_slice.device_ptr() as *mut u8;

        info!(
            "Output '{}': shape={:?}, size={} bytes",
            output_info.name, shape, size_bytes
        );

        device_outputs.push(DeviceTensor {
            name: output_info.name.clone(),
            device_ptr,
            size_bytes,
            shape,
            dtype: output_info.dtype.clone(),
        });

        device_slices.push(device_slice);
    }

    // Extended warmup to handle CUDA caching effects properly
    info!("Running {} warmup iterations to handle CUDA caching...", warmup);
    for i in 0..warmup {
        let _ = engine.pin_mut().infer(&cpu_inputs)?;
        engine
            .pin_mut()
            .infer_device(&device_inputs, &mut device_outputs)?;
        device.synchronize()?;  // Ensure each warmup completes
        
        // Log progress for long warmup
        if (i + 1) % 50 == 0 {
            info!("Warmup progress: {}/{}", i + 1, warmup);
        }
    }

    // Additional GPU boost clock stabilization
    info!("Allowing GPU boost clocks to stabilize...");
    std::thread::sleep(std::time::Duration::from_millis(1000));

    // Benchmark CPU tensors
    info!(
        "Benchmarking CPU tensor inference ({} iterations)...",
        benchmark_iterations
    );
    let mut cpu_latencies = Vec::with_capacity(benchmark_iterations);

    for i in 0..benchmark_iterations {
        let start = Instant::now();
        let _ = engine.pin_mut().infer(&cpu_inputs)?;
        // CPU path already includes implicit synchronization via H2D/D2H transfers
        cpu_latencies.push(start.elapsed());

        // Progress reporting every 100 iterations
        if (i + 1) % 100 == 0 {
            info!("CPU progress: {}/{}", i + 1, benchmark_iterations);
        }
    }

    // Benchmark device tensors
    info!(
        "Benchmarking device tensor inference ({} iterations)...",
        benchmark_iterations
    );
    let mut device_latencies = Vec::with_capacity(benchmark_iterations);

    for i in 0..benchmark_iterations {
        let start = Instant::now();
        engine
            .pin_mut()
            .infer_device(&device_inputs, &mut device_outputs)?;
        // CRITICAL: Synchronize after each inference to get actual completion time
        device.synchronize()?;
        device_latencies.push(start.elapsed());

        // Progress reporting every 100 iterations
        if (i + 1) % 100 == 0 {
            info!("Device progress: {}/{}", i + 1, benchmark_iterations);
        }
    }

    // Run correctness comparison
    info!(
        "Running {} correctness comparison iterations...",
        correctness_iterations
    );
    let mut max_diffs = Vec::new();

    for iteration in 1..=correctness_iterations {
        info!(
            "Correctness iteration {}/{}",
            iteration, correctness_iterations
        );

        // Run CPU and device tensor inference
        let cpu_outputs = engine.pin_mut().infer(&cpu_inputs)?;
        engine
            .pin_mut()
            .infer_device(&device_inputs, &mut device_outputs)?;

        // Copy device output data back to host for comparison (safe)
        let mut device_output_data = Vec::new();
        for (_device_output, device_slice) in device_outputs
            .iter()
            .zip(device_slices.iter().skip(device_inputs.len()))
        {
            let host_data = device.dtoh_sync_copy(device_slice)?;
            device_output_data.push(host_data);
        }

        // Compare outputs
        for (i, (cpu_output, device_data)) in cpu_outputs
            .iter()
            .zip(device_output_data.iter())
            .enumerate()
        {
            if cpu_output.data.len() != device_data.len() {
                warn!(
                    "Output {} size mismatch: CPU {} bytes, Device {} bytes",
                    i,
                    cpu_output.data.len(),
                    device_data.len()
                );
                continue;
            }

            // Simple comparison for FP32 data
            if let TensorDataType::FP32 = cpu_output.dtype {
                let cpu_f32: Vec<f32> = cpu_output
                    .data
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect();
                let device_f32: Vec<f32> = device_data
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect();

                let mut max_diff = 0.0f32;
                for (&cpu_val, &device_val) in cpu_f32.iter().zip(device_f32.iter()) {
                    let diff = (cpu_val - device_val).abs();
                    max_diff = max_diff.max(diff);
                }

                max_diffs.push(max_diff);
                info!("Output {} max absolute difference: {:.6}", i, max_diff);
            }
        }
    }

    // Calculate statistics
    let cpu_avg =
        cpu_latencies.iter().map(|t| t.as_secs_f64()).sum::<f64>() / cpu_latencies.len() as f64;
    let device_avg = device_latencies
        .iter()
        .map(|t| t.as_secs_f64())
        .sum::<f64>()
        / device_latencies.len() as f64;
    let speedup = cpu_avg / device_avg;

    // Calculate percentiles
    let mut cpu_sorted = cpu_latencies.clone();
    let mut device_sorted = device_latencies.clone();
    cpu_sorted.sort();
    device_sorted.sort();

    let cpu_p50 = cpu_sorted[cpu_sorted.len() / 2].as_secs_f64();
    let cpu_p95 = cpu_sorted[cpu_sorted.len() * 95 / 100].as_secs_f64();
    let cpu_p99 = cpu_sorted[cpu_sorted.len() * 99 / 100].as_secs_f64();

    let device_p50 = device_sorted[device_sorted.len() / 2].as_secs_f64();
    let device_p95 = device_sorted[device_sorted.len() * 95 / 100].as_secs_f64();
    let device_p99 = device_sorted[device_sorted.len() * 99 / 100].as_secs_f64();

    // Print comprehensive results
    info!("\n{}", "=".repeat(80));
    info!("COMPREHENSIVE DEVICE TENSOR COMPARISON RESULTS");
    info!("{}", "=".repeat(80));

    info!("Model Analysis:");
    info!("  TensorRT Engine: {}", args.engine.display());
    info!("  Device Index: {}", device_index);
    info!("  Random Seed: {}", seed);
    info!("  Tolerance: {:.6}", tolerance);
    info!("  Benchmark Iterations: {}", benchmark_iterations);
    info!("  Correctness Iterations: {}", correctness_iterations);

    info!("\nPERFORMANCE RESULTS:");
    info!("  CPU Average Latency: {:.6} sec", cpu_avg);
    info!("  CPU Throughput: {:.2} inferences/sec", 1.0 / cpu_avg);
    info!("  Device Average Latency: {:.6} sec", device_avg);
    info!(
        "  Device Throughput: {:.2} inferences/sec",
        1.0 / device_avg
    );
    info!("  Device Speedup: {:.2}x faster than CPU tensors", speedup);

    // Calculate memory transfer overhead
    let transfer_overhead = cpu_avg - device_avg;
    info!(
        "  Estimated memory transfer overhead: {:.6} seconds per inference",
        transfer_overhead
    );

    if transfer_overhead > 0.0 {
        let transfer_percentage = (transfer_overhead / cpu_avg) * 100.0;
        info!(
            "  Memory transfers account for {:.1}% of CPU tensor inference time",
            transfer_percentage
        );
    }

    info!("\nLATENCY PERCENTILES:");
    info!(
        "  CPU      - P50: {:.6}s, P95: {:.6}s, P99: {:.6}s",
        cpu_p50, cpu_p95, cpu_p99
    );
    info!(
        "  Device   - P50: {:.6}s, P95: {:.6}s, P99: {:.6}s",
        device_p50, device_p95, device_p99
    );

    info!("\nCORRECTNESS RESULTS:");
    let overall_max_diff = if !max_diffs.is_empty() {
        let max_diff = max_diffs.iter().fold(0.0f32, |a, &b| a.max(b));
        let avg_max_diff = max_diffs.iter().sum::<f32>() / max_diffs.len() as f32;

        info!(
            "  Maximum absolute difference across all outputs: {:.6}",
            max_diff
        );
        info!(
            "  Average maximum difference per output: {:.6}",
            avg_max_diff
        );

        if max_diff < tolerance {
            info!("  Status: CPU AND DEVICE OUTPUTS MATCH WITHIN TOLERANCE");
        } else {
            info!(
                "  Status: DIFFERENCES DETECTED (max: {:.6} > tolerance: {:.6})",
                max_diff, tolerance
            );
        }
        max_diff
    } else {
        info!("  No FP32 outputs found for comparison");
        0.0
    };

    info!("\n{}", "=".repeat(80));

    // Summary recommendation
    if overall_max_diff < tolerance {
        info!("OVERALL STATUS: CPU and device tensors produce consistent results");
        info!(
            "RECOMMENDATION: Device tensors provide {:.1}x speedup with good accuracy",
            speedup
        );
    } else {
        info!("OVERALL STATUS: CPU and device tensors have output differences");
        info!("RECOMMENDATION: Review device tensor implementation or increase tolerance");
    }

    Ok(())
}
