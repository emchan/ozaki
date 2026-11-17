import numpy as np
"""Apple Silicon backend dispatcher."""
def create_backend(backend, M, K, N, dtype=np.float32, model_dir="/tmp"):
    if backend == "cpu":
        from arch.apple.cpu.backend import create_backend as cb
    elif backend == "gpu":
        from arch.apple.gpu.backend import create_backend as cb
    elif backend == "npu":
        from arch.apple.npu.backend import create_backend as cb
    else:
        raise ValueError(f"unknown backend {backend}")
    return cb(backend, M, K, N, dtype, model_dir=model_dir)
