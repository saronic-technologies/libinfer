use approx::assert_relative_eq;
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use libinfer::{Engine, Options, TensorDataType};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn test_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("test")
}

fn cuda_ctx() -> Arc<CudaContext> {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    unsafe { ctx.disable_event_tracking() };
    ctx
}

fn load_engine(name: &str, ctx: &Arc<CudaContext>) -> Engine {
    let _ = ctx; // ensure CUDA context is current
    let path = test_dir().join(name);
    let options = Options::new(path.to_string_lossy().to_string(), None);
    Engine::new(&options).expect("failed to load engine")
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

// --- Shared logger ---

#[test]
fn test_multiple_engines_no_duplicate_logger_warning() {
    use std::io::Read;
    use std::os::unix::io::FromRawFd;

    let mut fds = [0i32; 2];
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    let read_fd = fds[0];
    let write_fd = fds[1];

    let old_stderr = unsafe { libc::dup(2) };
    assert!(old_stderr >= 0);
    assert_eq!(unsafe { libc::dup2(write_fd, 2) }, 2);

    let ctx = cuda_ctx();
    let _engine1 = load_engine("test_dynamic.engine", &ctx);
    let _engine2 = load_engine("test_multi_input.engine", &ctx);
    let _engine3 = load_engine("test_dynamic.engine", &ctx);

    assert_eq!(unsafe { libc::dup2(old_stderr, 2) }, 2);
    unsafe {
        libc::close(old_stderr);
        libc::close(write_fd);
    }

    let mut captured = String::new();
    let mut reader = unsafe { std::fs::File::from_raw_fd(read_fd) };
    reader.read_to_string(&mut captured).ok();

    assert!(
        !captured.contains("logger passed into createInferRuntime differs"),
        "duplicate logger warning detected in stderr:\n{captured}"
    );
}

// --- Error handling ---

#[test]
fn test_load_nonexistent_path() {
    let ctx = cuda_ctx();
    let _ = &ctx;
    let options = Options::new("/nonexistent/path/model.engine".to_string(), None);
    assert!(Engine::new(&options).is_err());
}

#[test]
fn test_load_invalid_engine() {
    let ctx = cuda_ctx();
    let _ = &ctx;
    let path = test_dir().join("test_dynamic.onnx"); // valid file, not a TRT engine
    let options = Options::new(path.to_string_lossy().to_string(), None);
    assert!(Engine::new(&options).is_err());
}

#[test]
fn test_wrong_num_inputs() {
    let ctx = cuda_ctx();
    let mut engine = load_engine("test_multi_input.engine", &ctx);
    let stream = ctx.new_stream().expect("failed to create stream");

    let buf = stream.alloc_zeros::<u8>(3 * 4).expect("alloc failed");
    let mut out_buf = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");

    let (ip, _) = buf.device_ptr(&stream);
    let (op, _) = out_buf.device_ptr_mut(&stream);

    // engine expects 2 inputs, pass 1
    let result = engine.infer(&[ip], &[op], stream.cu_stream(), 1);
    assert!(result.is_err());
}

#[test]
fn test_wrong_num_outputs() {
    let ctx = cuda_ctx();
    let mut engine = load_engine("test_dynamic.engine", &ctx);
    let stream = ctx.new_stream().expect("failed to create stream");

    let buf = stream.alloc_zeros::<u8>(4 * 4).expect("alloc failed");
    let mut out_a = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");
    let mut out_b = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");

    let (ip, _) = buf.device_ptr(&stream);
    let (op_a, _) = out_a.device_ptr_mut(&stream);
    let (op_b, _) = out_b.device_ptr_mut(&stream);

    // engine expects 1 output, pass 2
    let result = engine.infer(&[ip], &[op_a, op_b], stream.cu_stream(), 1);
    assert!(result.is_err());
}

#[test]
fn test_batch_size_below_min() {
    let ctx = cuda_ctx();
    let mut engine = load_engine("test_dynamic.engine", &ctx);
    let stream = ctx.new_stream().expect("failed to create stream");

    let buf = stream.alloc_zeros::<u8>(4 * 4).expect("alloc failed");
    let mut out_buf = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");

    let (ip, _) = buf.device_ptr(&stream);
    let (op, _) = out_buf.device_ptr_mut(&stream);

    // min batch is 1, pass 0
    let result = engine.infer(&[ip], &[op], stream.cu_stream(), 0);
    assert!(result.is_err());
}

#[test]
fn test_batch_size_above_max() {
    let ctx = cuda_ctx();
    let mut engine = load_engine("test_dynamic.engine", &ctx);
    let stream = ctx.new_stream().expect("failed to create stream");

    let buf = stream.alloc_zeros::<u8>(9 * 4 * 4).expect("alloc failed");
    let mut out_buf = stream.alloc_zeros::<u8>(9 * 2 * 4).expect("alloc failed");

    let (ip, _) = buf.device_ptr(&stream);
    let (op, _) = out_buf.device_ptr_mut(&stream);

    // max batch is 8, pass 9
    let result = engine.infer(&[ip], &[op], stream.cu_stream(), 9);
    assert!(result.is_err());
}

// --- Dynamic batch tests (test_dynamic.engine) ---
// Model: output = relu(input @ W + B)
// W = [[1,0],[0,1],[1,1],[0,0]], B = [0.5, -0.5]
// input [1,2,3,4] -> matmul [4,5] -> add [4.5, 4.5] -> relu [4.5, 4.5]

#[test]
fn test_dynamic_batch_dims() {
    let ctx = cuda_ctx();
    let engine = load_engine("test_dynamic.engine", &ctx);
    let batch = engine.get_batch_dims();

    assert_eq!(batch.min, 1);
    assert_eq!(batch.opt, 4);
    assert_eq!(batch.max, 8);

    assert_eq!(engine.get_num_inputs(), 1);
    assert_eq!(engine.get_num_outputs(), 1);

    let inputs = engine.get_input_dims();
    assert_eq!(inputs[0].name, "input");
    assert_eq!(inputs[0].dims, &[4]);
    assert_eq!(inputs[0].dtype, TensorDataType::FP32);
    assert_eq!(inputs[0].byte_size(), 4 * 4);

    let outputs = engine.get_output_dims();
    assert_eq!(outputs[0].name, "output");
    assert_eq!(outputs[0].dims, &[2]);
    assert_eq!(outputs[0].dtype, TensorDataType::FP32);
    assert_eq!(outputs[0].byte_size(), 2 * 4);
    assert_eq!(engine.get_output_len(), 2);
}

#[test]
fn test_dynamic_batch_single() {
    let ctx = cuda_ctx();
    let mut engine = load_engine("test_dynamic.engine", &ctx);
    let stream = ctx.new_stream().expect("failed to create stream");

    // batch=1, input=[1, 2, 3, 4] -> expected output=[4.5, 4.5]
    let input = f32_to_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let input_buf = stream.clone_htod(&input).expect("H2D failed");
    let mut output_buf = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");

    {
        let (ip, _) = input_buf.device_ptr(&stream);
        let (op, _) = output_buf.device_ptr_mut(&stream);
        engine
            .infer(&[ip], &[op], stream.cu_stream(), 1)
            .expect("inference failed");
    }

    let output = bytes_to_f32(&stream.clone_dtoh(&output_buf).expect("D2H failed"));
    assert_relative_eq!(output[0], 4.5, epsilon = 1e-3);
    assert_relative_eq!(output[1], 4.5, epsilon = 1e-3);
}

#[test]
fn test_dynamic_batch_multiple() {
    let ctx = cuda_ctx();
    let mut engine = load_engine("test_dynamic.engine", &ctx);
    let stream = ctx.new_stream().expect("failed to create stream");

    // batch=3: three copies of [1, 2, 3, 4] -> each should produce [4.5, 4.5]
    let single = [1.0f32, 2.0, 3.0, 4.0];
    let input = f32_to_bytes(&single.repeat(3));
    let input_buf = stream.clone_htod(&input).expect("H2D failed");
    let mut output_buf = stream.alloc_zeros::<u8>(3 * 2 * 4).expect("alloc failed");

    {
        let (ip, _) = input_buf.device_ptr(&stream);
        let (op, _) = output_buf.device_ptr_mut(&stream);
        engine
            .infer(&[ip], &[op], stream.cu_stream(), 3)
            .expect("inference failed");
    }

    let output = bytes_to_f32(&stream.clone_dtoh(&output_buf).expect("D2H failed"));
    assert_eq!(output.len(), 6);
    for chunk in output.chunks_exact(2) {
        assert_relative_eq!(chunk[0], 4.5, epsilon = 1e-3);
        assert_relative_eq!(chunk[1], 4.5, epsilon = 1e-3);
    }
}

// --- Multi-input tests (test_multi_input.engine) ---
// Model: output = concat(input_a, input_b) @ W
// W sums first 3 elements into output[0], last 5 into output[1]
// input_a=[1,2,3], input_b=[1,1,1,1,1] -> output=[6, 5]

#[test]
fn test_multi_input_metadata() {
    let ctx = cuda_ctx();
    let engine = load_engine("test_multi_input.engine", &ctx);

    assert_eq!(engine.get_num_inputs(), 2);
    assert_eq!(engine.get_num_outputs(), 1);

    let inputs = engine.get_input_dims();
    assert_eq!(inputs[0].name, "input_a");
    assert_eq!(inputs[0].dims, &[3]);
    assert_eq!(inputs[0].dtype, TensorDataType::FP32);
    assert_eq!(inputs[1].name, "input_b");
    assert_eq!(inputs[1].dims, &[5]);
    assert_eq!(inputs[1].dtype, TensorDataType::FP32);

    let outputs = engine.get_output_dims();
    assert_eq!(outputs[0].name, "output");
    assert_eq!(outputs[0].dims, &[2]);
    assert_eq!(outputs[0].dtype, TensorDataType::FP32);
    assert_eq!(engine.get_output_len(), 2);

    let batch = engine.get_batch_dims();
    assert_eq!(batch.min, 1);
    assert_eq!(batch.opt, 4);
    assert_eq!(batch.max, 8);
}

#[test]
fn test_multi_input_inference() {
    let ctx = cuda_ctx();
    let mut engine = load_engine("test_multi_input.engine", &ctx);
    let stream = ctx.new_stream().expect("failed to create stream");

    // input_a=[1,2,3], input_b=[1,1,1,1,1] -> output=[6, 5]
    let input_a = f32_to_bytes(&[1.0, 2.0, 3.0]);
    let input_b = f32_to_bytes(&[1.0, 1.0, 1.0, 1.0, 1.0]);

    let buf_a = stream.clone_htod(&input_a).expect("H2D failed");
    let buf_b = stream.clone_htod(&input_b).expect("H2D failed");
    let mut output_buf = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");

    {
        let (pa, _) = buf_a.device_ptr(&stream);
        let (pb, _) = buf_b.device_ptr(&stream);
        let (op, _) = output_buf.device_ptr_mut(&stream);
        engine
            .infer(&[pa, pb], &[op], stream.cu_stream(), 1)
            .expect("inference failed");
    }

    let output = bytes_to_f32(&stream.clone_dtoh(&output_buf).expect("D2H failed"));
    assert_relative_eq!(output[0], 6.0, epsilon = 1e-3);
    assert_relative_eq!(output[1], 5.0, epsilon = 1e-3);
}

#[test]
fn test_multi_input_batch() {
    let ctx = cuda_ctx();
    let mut engine = load_engine("test_multi_input.engine", &ctx);
    let stream = ctx.new_stream().expect("failed to create stream");

    // batch=2:
    // row 0: input_a=[1,2,3], input_b=[1,1,1,1,1] -> [6, 5]
    // row 1: input_a=[0,0,0], input_b=[2,2,2,2,2] -> [0, 10]
    let input_a = f32_to_bytes(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    let input_b = f32_to_bytes(&[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]);

    let buf_a = stream.clone_htod(&input_a).expect("H2D failed");
    let buf_b = stream.clone_htod(&input_b).expect("H2D failed");
    let mut output_buf = stream.alloc_zeros::<u8>(2 * 2 * 4).expect("alloc failed");

    {
        let (pa, _) = buf_a.device_ptr(&stream);
        let (pb, _) = buf_b.device_ptr(&stream);
        let (op, _) = output_buf.device_ptr_mut(&stream);
        engine
            .infer(&[pa, pb], &[op], stream.cu_stream(), 2)
            .expect("inference failed");
    }

    let output = bytes_to_f32(&stream.clone_dtoh(&output_buf).expect("D2H failed"));
    assert_eq!(output.len(), 4);
    // row 0
    assert_relative_eq!(output[0], 6.0, epsilon = 1e-3);
    assert_relative_eq!(output[1], 5.0, epsilon = 1e-3);
    // row 1
    assert_relative_eq!(output[2], 0.0, epsilon = 1e-3);
    assert_relative_eq!(output[3], 10.0, epsilon = 1e-3);
}

// --- CUDA graph tests ---
// These tests prove that graph replay produces bit-identical results to
// the normal enqueueV3 path, and that pointer changes trigger recapture
// rather than silently writing to stale addresses.

fn load_engine_with_graphs(name: &str, ctx: &Arc<CudaContext>) -> Engine {
    let _ = ctx;
    let path = test_dir().join(name);
    let options = Options::new(path.to_string_lossy().to_string(), Some(0));
    Engine::new(&options).expect("failed to load engine")
}

/// Run the same input through both the non-graph and graph paths and
/// assert the outputs are identical. The graph engine runs twice: once
/// to capture (cold), once to replay (hot). Both must match the baseline.
#[test]
fn test_cuda_graph_matches_baseline() {
    let ctx = cuda_ctx();
    let stream = ctx.new_stream().expect("failed to create stream");

    let input = f32_to_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let input_buf = stream.clone_htod(&input).expect("H2D failed");

    // Baseline: no graphs
    let mut baseline_engine = load_engine("test_dynamic.engine", &ctx);
    let mut baseline_out = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");
    {
        let (ip, _) = input_buf.device_ptr(&stream);
        let (op, _) = baseline_out.device_ptr_mut(&stream);
        baseline_engine
            .infer(&[ip], &[op], stream.cu_stream(), 1)
            .expect("baseline inference failed");
    }
    let baseline = bytes_to_f32(&stream.clone_dtoh(&baseline_out).expect("D2H failed"));

    // Graph path: capture pass then replay pass
    let mut graph_engine = load_engine_with_graphs("test_dynamic.engine", &ctx);
    let mut graph_out = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");
    let (ip, _) = input_buf.device_ptr(&stream);
    let (op, _) = graph_out.device_ptr_mut(&stream);

    for pass in ["capture", "replay"] {
        graph_engine
            .infer(&[ip], &[op], stream.cu_stream(), 1)
            .unwrap_or_else(|e| panic!("{pass} failed: {e}"));

        let result = bytes_to_f32(&stream.clone_dtoh(&graph_out).expect("D2H failed"));
        assert_eq!(
            result, baseline,
            "graph {pass} output differs from baseline: {result:?} vs {baseline:?}"
        );
    }
}

/// Same as above but with batch=3 — proves the graph captures the
/// correct batch dimension, not just batch=1.
#[test]
fn test_cuda_graph_matches_baseline_batched() {
    let ctx = cuda_ctx();
    let stream = ctx.new_stream().expect("failed to create stream");

    let single = [1.0f32, 2.0, 3.0, 4.0];
    let input = f32_to_bytes(&single.repeat(3));
    let input_buf = stream.clone_htod(&input).expect("H2D failed");

    let mut baseline_engine = load_engine("test_dynamic.engine", &ctx);
    let mut baseline_out = stream.alloc_zeros::<u8>(3 * 2 * 4).expect("alloc failed");
    {
        let (ip, _) = input_buf.device_ptr(&stream);
        let (op, _) = baseline_out.device_ptr_mut(&stream);
        baseline_engine
            .infer(&[ip], &[op], stream.cu_stream(), 3)
            .expect("baseline failed");
    }
    let baseline = bytes_to_f32(&stream.clone_dtoh(&baseline_out).expect("D2H failed"));

    let mut graph_engine = load_engine_with_graphs("test_dynamic.engine", &ctx);
    let mut graph_out = stream.alloc_zeros::<u8>(3 * 2 * 4).expect("alloc failed");
    let (ip, _) = input_buf.device_ptr(&stream);
    let (op, _) = graph_out.device_ptr_mut(&stream);

    for pass in ["capture", "replay"] {
        graph_engine
            .infer(&[ip], &[op], stream.cu_stream(), 3)
            .unwrap_or_else(|e| panic!("{pass} failed: {e}"));

        let result = bytes_to_f32(&stream.clone_dtoh(&graph_out).expect("D2H failed"));
        assert_eq!(
            result, baseline,
            "graph {pass} (batch=3) differs from baseline: {result:?} vs {baseline:?}"
        );
    }
}

/// Runs batch sizes 1, 4, 8 then repeats them. The second pass hits
/// cached graphs. Each result is compared against the non-graph baseline
/// for that batch size.
#[test]
fn test_cuda_graph_multiple_batch_sizes() {
    let ctx = cuda_ctx();
    let stream = ctx.new_stream().expect("failed to create stream");

    let single = [1.0f32, 2.0, 3.0, 4.0];
    let input = f32_to_bytes(&single.repeat(8));
    let input_buf = stream.clone_htod(&input).expect("H2D failed");

    // Collect baselines for each batch size
    let mut baseline_engine = load_engine("test_dynamic.engine", &ctx);
    let mut baseline_out = stream.alloc_zeros::<u8>(8 * 2 * 4).expect("alloc failed");
    let (ip, _) = input_buf.device_ptr(&stream);
    let (bop, _) = baseline_out.device_ptr_mut(&stream);

    let mut baselines = std::collections::HashMap::new();
    for &bs in &[1u32, 4, 8] {
        baseline_engine
            .infer(&[ip], &[bop], stream.cu_stream(), bs)
            .expect("baseline failed");
        let out = bytes_to_f32(&stream.clone_dtoh(&baseline_out).expect("D2H failed"));
        baselines.insert(bs, out[..bs as usize * 2].to_vec());
    }

    // Graph engine: first pass captures, second replays
    let mut graph_engine = load_engine_with_graphs("test_dynamic.engine", &ctx);
    let mut graph_out = stream.alloc_zeros::<u8>(8 * 2 * 4).expect("alloc failed");
    let (gop, _) = graph_out.device_ptr_mut(&stream);

    for &bs in &[1u32, 4, 8, 1, 4, 8] {
        graph_engine
            .infer(&[ip], &[gop], stream.cu_stream(), bs)
            .unwrap_or_else(|e| panic!("batch_size={bs} failed: {e}"));

        let result = bytes_to_f32(&stream.clone_dtoh(&graph_out).expect("D2H failed"));
        let result = &result[..bs as usize * 2];
        let expected = &baselines[&bs];
        assert_eq!(
            result, expected.as_slice(),
            "batch_size={bs} graph output differs from baseline"
        );
    }
}

/// Proves pointer validation works: allocate new buffers (different
/// device addresses), run inference. If the graph wasn't recaptured,
/// it would write to the old output buffer and the new one would
/// remain zeroed.
#[test]
fn test_cuda_graph_pointer_invalidation() {
    let ctx = cuda_ctx();
    let mut engine = load_engine_with_graphs("test_dynamic.engine", &ctx);
    let stream = ctx.new_stream().expect("failed to create stream");

    let input_data = f32_to_bytes(&[1.0, 2.0, 3.0, 4.0]);

    // First allocation — captures graph targeting buf1
    let input_buf1 = stream.clone_htod(&input_data).expect("H2D failed");
    let mut output_buf1 = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");
    {
        let (ip, _) = input_buf1.device_ptr(&stream);
        let (op, _) = output_buf1.device_ptr_mut(&stream);
        engine
            .infer(&[ip], &[op], stream.cu_stream(), 1)
            .expect("capture failed");
    }

    // Second allocation — different device addresses
    let input_buf2 = stream.clone_htod(&input_data).expect("H2D failed");
    let mut output_buf2 = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");

    // Verify output_buf2 starts at zero (so a stale graph would be caught)
    let zeros = bytes_to_f32(&stream.clone_dtoh(&output_buf2).expect("D2H failed"));
    assert_eq!(zeros, vec![0.0, 0.0], "output_buf2 should start zeroed");

    {
        let (ip, _) = input_buf2.device_ptr(&stream);
        let (op, _) = output_buf2.device_ptr_mut(&stream);
        engine
            .infer(&[ip], &[op], stream.cu_stream(), 1)
            .expect("recapture failed");
    }

    // If recapture happened, output_buf2 has the correct result.
    // If it didn't, the graph wrote to output_buf1 and buf2 is still zero.
    let result = bytes_to_f32(&stream.clone_dtoh(&output_buf2).expect("D2H failed"));
    assert_relative_eq!(result[0], 4.5, epsilon = 1e-3);
    assert_relative_eq!(result[1], 4.5, epsilon = 1e-3);
}

/// Multi-input model: verifies graph capture/replay with two input
/// tensors by comparing against the non-graph baseline.
#[test]
fn test_cuda_graph_multi_input() {
    let ctx = cuda_ctx();
    let stream = ctx.new_stream().expect("failed to create stream");

    let input_a = f32_to_bytes(&[1.0, 2.0, 3.0]);
    let input_b = f32_to_bytes(&[1.0, 1.0, 1.0, 1.0, 1.0]);
    let buf_a = stream.clone_htod(&input_a).expect("H2D failed");
    let buf_b = stream.clone_htod(&input_b).expect("H2D failed");

    // Baseline
    let mut baseline_engine = load_engine("test_multi_input.engine", &ctx);
    let mut baseline_out = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");
    {
        let (pa, _) = buf_a.device_ptr(&stream);
        let (pb, _) = buf_b.device_ptr(&stream);
        let (op, _) = baseline_out.device_ptr_mut(&stream);
        baseline_engine
            .infer(&[pa, pb], &[op], stream.cu_stream(), 1)
            .expect("baseline failed");
    }
    let baseline = bytes_to_f32(&stream.clone_dtoh(&baseline_out).expect("D2H failed"));

    // Graph path
    let mut graph_engine = load_engine_with_graphs("test_multi_input.engine", &ctx);
    let mut graph_out = stream.alloc_zeros::<u8>(2 * 4).expect("alloc failed");
    let (pa, _) = buf_a.device_ptr(&stream);
    let (pb, _) = buf_b.device_ptr(&stream);
    let (op, _) = graph_out.device_ptr_mut(&stream);

    for pass in ["capture", "replay"] {
        graph_engine
            .infer(&[pa, pb], &[op], stream.cu_stream(), 1)
            .unwrap_or_else(|e| panic!("{pass} failed: {e}"));

        let result = bytes_to_f32(&stream.clone_dtoh(&graph_out).expect("D2H failed"));
        assert_eq!(
            result, baseline,
            "graph {pass} (multi-input) differs from baseline: {result:?} vs {baseline:?}"
        );
    }
}
