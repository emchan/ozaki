"""AMD Radeon iGPU backend: ONNX Runtime DirectMLExecutionProvider."""
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto


def _dtype_proto(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.float16:
        return TensorProto.FLOAT16
    if dtype == np.float32:
        return TensorProto.FLOAT
    raise ValueError(f"Unsupported dtype {dtype}")


class Backend:
    def __init__(self, backend, M, K, N, dtype, model_dir=None):
        if backend != "gpu":
            raise ValueError(f"gpu backend expected, got {backend}")
        if model_dir is None:
            import tempfile
            model_dir = "/tmp/ozaki"
        self.backend = backend
        self.M, self.K, self.N = M, K, N
        self.dtype = np.dtype(dtype)
        self.proto = _dtype_proto(self.dtype)
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"ozaki_backend_gpu_{M}_{K}_{N}_{self.dtype}.onnx")
        A = helper.make_tensor_value_info("A", self.proto, [M, K])
        B = helper.make_tensor_value_info("B", self.proto, [K, N])
        C = helper.make_tensor_value_info("C", self.proto, [M, N])
        node = helper.make_node("MatMul", ["A", "B"], ["C"])
        g = helper.make_graph([node], "gemm", [A, B], [C])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 9
        onnx.save(m, path)
        self.path = path
        self.sess = ort.InferenceSession(path, providers=["DmlExecutionProvider"])

    def run(self, A_, B_):
        Aa = np.asarray(A_, dtype=self.dtype)
        Bb = np.asarray(B_, dtype=self.dtype)
        out = self.sess.run(None, {"A": Aa, "B": Bb})[0]
        return np.array(out, dtype=self.dtype)


def create_backend(backend, M, K, N, dtype=np.float32, model_dir=None):
    return Backend(backend, M, K, N, dtype, model_dir=model_dir)
