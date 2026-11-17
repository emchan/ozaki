import os, sys, time
import numpy as np
os.environ["RYZEN_AI_INSTALLATION_PATH"] = r"C:\Program Files\RyzenAI\1.9.0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ozaki as O

def get_backend_factory(backend):
    if backend == 'npu':
        from arch.amd.npu.backend import create_backend as cb
    elif backend == 'gpu':
        from arch.amd.gpu.backend import create_backend as cb
    else:
        from arch.amd.cpu.backend import create_backend as cb
    return cb

K, N, T, M = 3072, 12288, 1024, 1
A64 = (np.random.standard_normal((M * T, K)) * 0.5).astype(np.float64)
B64 = (np.random.standard_normal((K, N)) * 0.5).astype(np.float64)
ref = A64 @ B64
A_t = A64.astype(np.float64); B_t = B64.astype(np.float64)

for backend in ("cpu", "gpu", "npu"):
    cb = get_backend_factory(backend)
    for s in (2, 3, 4, 6):
        try:
            sess = cb(backend, M*T, K, N, np.float32)
            ts = []
            for _ in range(1):
                t0 = time.perf_counter()
                C, _ = O.ozaki_matmul(A_t, B_t, s, backend, np.float32, np.float64, 2**-24, session=sess)
                ts.append(time.perf_counter() - t0)
            C = np.asarray(C, dtype=np.float64)
            err = float(np.max(np.abs(C - ref)))
            n_partials = s * (s + 1) // 2
            print(f"{backend:4s} ozaki fp64 low=fp32 s={s} ({n_partials} partials): {min(ts)*1000:9.2f} ms  maxabs_vs_fp64 {err:.3e}", flush=True)
        except Exception as e:
            print(f"{backend:4s} s={s}: ERROR {e}")
