"""Snapdragon backend: integrated CPU (ARM Oryon) via ONNX Runtime CPUExecutionProvider.

Project layout:
    ozaki/
      common/            -> the ozaki method core (splitting, matmul, session wrapper)
      arch/snapdragon/   -> architecture-specific backends
        cpu/              -> this module
        gpu/              -> Adreno (QNN EP) backend
        npu/              -> Hexagon (QNN EP) backend
      bench/             -> experiment / benchmark scripts
      run_tests.py       -> scheme harness
"""
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto


def create_backend(backend: str, M: int, K: int, N: int, dtype=None, model_dir: str = "/tmp/ozaki"):
    """Return an object with .run(A, B) -> C for `A @ B` with A:(M,K) B:(K,N)."""
    if backend == "cpu":
        dt = {np.float32: TensorProto.FLOAT, np.float16: TensorProto.FLOAT16}[dtype]
        A = helper.make_tensor_value_info("A", dt, [M, K])
        B = helper.make_tensor_value_info("B", dt, [K, N])
        C = helper.make_tensor_value_info("C", dt, [M, N])
        node = helper.make_node("MatMul", ["A", "B"], ["C"])
        g = helper.make_graph([node], "gemm", [A, B], [C])
        m = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid("", 13)])
        m.ir_version = 9
        path = os.path.join(model_dir, f"ozaki_backend_cpu_{M}_{K}_{N}_{str(dtype)}.onnx")
        onnx.save(m, path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

        class _Backend:
            def run(self, A_, B_):
                return np.array(sess.run(None, {"A": np.asarray(A_, dtype=dtype), "B": np.asarray(B_, dtype=dtype)})[0])

        return _Backend()
    if backend in ("npu", "gpu"):
        from common.ozaki import QNNSession

        return QNNSession(M, K, N, dtype, backend)
    raise ValueError(f"unknown backend {backend}")
