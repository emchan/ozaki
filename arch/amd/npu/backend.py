"""AMD Ryzen NPU backend via VitisAI Execution Provider (Ryzen AI Software)."""
import os, sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

ROOT = os.environ.get("RYZEN_AI_INSTALLATION_PATH",
                      r"C:\Program Files\RyzenAI\1.9.0")

def _register():
    dll = os.path.join(ROOT, "onnxruntime", "bin", "onnxruntime_providers_vitisai.dll")
    if os.path.exists(dll):
        try:
            ort.register_execution_provider_library("VitisAIExecutionProvider", dll)
        except Exception:
            pass
    # ensure PATH
    for p in [os.path.join(ROOT, "onnxruntime", "bin"),
              os.path.join(ROOT, "voe-4.0-win_amd64", "bin"),
              r"C:\Windows\System32\AMD"]:
        if p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")

_register()

def _dtype_proto(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.float16:
        return TensorProto.FLOAT16
    if dtype == np.float32:
        return TensorProto.FLOAT
    raise ValueError(f"Unsupported dtype {dtype}")

class Backend:
    def __init__(self, backend, M, K, N, dtype, model_dir=None):
        if backend != "npu":
            raise ValueError("npu backend expected")
        if model_dir is None:
            import tempfile
            model_dir = tempfile.gettempdir()
        os.makedirs(model_dir, exist_ok=True)
        self.backend = backend
        self.M, self.K, self.N = M, K, N
        self.dtype = np.dtype(dtype)
        self.proto = _dtype_proto(self.dtype)
        path = os.path.join(model_dir, f"ozaki_backend_npu_{M}_{K}_{N}_{self.dtype}.onnx")
        A = helper.make_tensor_value_info("A", self.proto, [M, K])
        B = helper.make_tensor_value_info("B", self.proto, [K, N])
        C = helper.make_tensor_value_info("C", self.proto, [M, N])
        node = helper.make_node("MatMul", ["A", "B"], ["C"])
        g = helper.make_graph([node], "gemm", [A, B], [C])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        m.ir_version = 9
        onnx.save(m, path)
        self.path = path
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, so, providers=["VitisAIExecutionProvider"], provider_options=[{}])

    def run(self, A_, B_):
        Aa = np.asarray(A_, dtype=self.dtype)
        Bb = np.asarray(B_, dtype=self.dtype)
        out = self.sess.run(None, {"A": Aa, "B": Bb})[0]
        return np.array(out, dtype=self.dtype)

def create_backend(backend, M, K, N, dtype=np.float32, model_dir=None):
    return Backend(backend, M, K, N, dtype, model_dir=model_dir)
