"""macOS Apple Silicon NPU backend (CoreML / Apple Neural Engine)."""
import time
import numpy as np

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types as mt
    _COREML_AVAILABLE = True
except Exception:  # pragma: no cover - platform dependent
    ct = None
    mb = None
    mt = None
    _COREML_AVAILABLE = False


def _mil_dtype(npdt):
    if not _COREML_AVAILABLE:
        raise RuntimeError("coremltools is required for the NPU backend")
    if npdt == np.float16:
        return mt.fp16
    if npdt == np.float32:
        return mt.fp32
    raise ValueError(f"Unsupported dtype for CoreML MIL: {npdt}")


class Backend:
    def __init__(self, backend, M, K, N, dtype, model_dir="/tmp/ozaki"):
        if backend != "npu":
            raise ValueError(f"npu backend expected, got {backend}")
        if not _COREML_AVAILABLE:
            raise RuntimeError("coremltools is required for the NPU backend")
        self.backend = backend
        self.M, self.K, self.N = M, K, N
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float16, np.float32):
            raise ValueError(f"Unsupported dtype for Apple backends: {self.dtype}")
        mil_dt = _mil_dtype(self.dtype)

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(M, K), dtype=mil_dt),
            mb.TensorSpec(shape=(K, N), dtype=mil_dt),
        ])
        def prog(a, b):
            return mb.matmul(x=a, y=b, transpose_x=False, transpose_y=False)

        self.model = ct.convert(
            prog,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.macOS13,
        )

    def run(self, A_, B_):
        Aa = np.asarray(A_, dtype=self.dtype)
        Bb = np.asarray(B_, dtype=self.dtype)
        res = self.model.predict({
            "a": Aa.astype(np.float32),
            "b": Bb.astype(np.float32),
        })
        out = np.asarray(list(res.values())[0], dtype=np.float32)
        return np.array(out, dtype=self.dtype)


def create_backend(backend, M, K, N, dtype=np.float32, model_dir="/tmp/ozaki"):
    return Backend(backend, M, K, N, dtype, model_dir=model_dir)
