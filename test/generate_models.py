#!/usr/bin/env python3
"""Generate dummy ONNX models for integration testing.

Produces two models with known weights so that inference outputs are
deterministic and verifiable:

  test_dynamic.onnx      - single input (batch, 4), single output (batch, 2), dynamic batch
  test_multi_input.onnx  - two inputs (batch, 3) + (batch, 5), single output (batch, 2), dynamic batch

Build TensorRT engines from these with trtexec:

  trtexec --onnx=test/test_dynamic.onnx --saveEngine=test/test_dynamic.engine \
          --minShapes=input:1x4 --optShapes=input:4x4 --maxShapes=input:8x4

  trtexec --onnx=test/test_multi_input.onnx --saveEngine=test/test_multi_input.engine \
          --minShapes=input_a:1x3,input_b:1x5 --optShapes=input_a:4x3,input_b:4x5 \
          --maxShapes=input_a:8x3,input_b:8x5
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SCRIPT_DIR = Path(__file__).parent


def make_dynamic_model():
    """Single input, single output, dynamic batch.

    Graph: output = relu(input @ W + B)
    Input: (batch, 4) Output: (batch, 2)

    W is [[1, 0],    B is [0.5, -0.5]
          [0, 1],
          [1, 1],
          [0, 0]]

    So for input [1, 2, 3, 4]:
      matmul = [1*1+2*0+3*1+4*0, 1*0+2*1+3*1+4*0] = [4, 5]
      add    = [4.5, 4.5]
      relu   = [4.5, 4.5]
    """
    W = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
    B = np.array([0.5, -0.5], dtype=np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "W"], ["matmul_out"]),
            helper.make_node("Add", ["matmul_out", "B"], ["add_out"]),
            helper.make_node("Relu", ["add_out"], ["output"]),
        ],
        "dynamic_batch",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
        initializer=[
            numpy_helper.from_array(W, name="W"),
            numpy_helper.from_array(B, name="B"),
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path = SCRIPT_DIR / "test_dynamic.onnx"
    onnx.save(model, str(path))
    print(f"wrote {path}")


def make_multi_input_model():
    """Two inputs concatenated, single output, dynamic batch.

    Graph: output = concat(input_a, input_b) @ W
    input_a: (batch, 3), input_b: (batch, 5) -> concat -> (batch, 8) -> matmul -> (batch, 2)

    W is [[1, 0],    (8x2 - sums first 3 and last 5 elements separately)
          [1, 0],
          [1, 0],
          [0, 1],
          [0, 1],
          [0, 1],
          [0, 1],
          [0, 1]]

    So for input_a=[1,2,3], input_b=[1,1,1,1,1]:
      concat  = [1,2,3,1,1,1,1,1]
      output  = [1+2+3, 1+1+1+1+1] = [6, 5]
    """
    W = np.zeros((8, 2), dtype=np.float32)
    W[:3, 0] = 1.0  # first 3 rows sum into output[0]
    W[3:, 1] = 1.0  # last 5 rows sum into output[1]

    graph = helper.make_graph(
        [
            helper.make_node("Concat", ["input_a", "input_b"], ["concat_out"], axis=1),
            helper.make_node("MatMul", ["concat_out", "W"], ["output"]),
        ],
        "multi_input",
        [
            helper.make_tensor_value_info(
                "input_a", TensorProto.FLOAT, ["batch", 3]
            ),
            helper.make_tensor_value_info(
                "input_b", TensorProto.FLOAT, ["batch", 5]
            ),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
        initializer=[numpy_helper.from_array(W, name="W")],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path = SCRIPT_DIR / "test_multi_input.onnx"
    onnx.save(model, str(path))
    print(f"wrote {path}")


if __name__ == "__main__":
    make_dynamic_model()
    make_multi_input_model()
