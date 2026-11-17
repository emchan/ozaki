import os, sys, time
import numpy as np
if len(sys.argv) > 1 and sys.argv[1] == 'npu':
    os.environ["RYZEN_AI_INSTALLATION_PATH"] = r"C:\Program Files\RyzenAI\1.9.0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ozaki as O

backend = sys.argv[1] if len(sys.argv) > 1 else 'cpu'

def get_cb(backend):
    if backend == 'npu':
        from arch.amd.npu.backend import create_backend as cb
    elif backend == 'gpu':
        from arch.amd.gpu.backend import create_backend as cb
    else:
        from arch.amd.cpu.backend import create_backend as cb
    return cb

cb = get_cb(backend)

K, N, T, M = 3072, 12288, 1024, 1
A64 = (np.random.standard_normal((M * T, K)) * 0.5).astype(np.float64)
B64 = (np.random.standard_normal((K, N)) * 0.5).astype(np.float64)
ref = A64 @ B64
A_t = A64.astype(np.float32); B_t = B64.astype(np.float32)

s=8; u=2**-11
sess = cb(backend, M*T, K, N, np.float16)
ts = []
for _ in range(1):
    t0 = time.perf_counter()
    C, _ = O.ozaki_matmul(A_t, B_t, s, backend, np.float16, np.float32, u, session=sess)
    ts.append(time.perf_counter() - t0)
C = np.asarray(C, dtype=np.float32)
err = float(np.max(np.abs(C.astype(np.float64) - ref)))
print(f"{backend:4s} ozaki fp32_emu (low=fp16, s=8, u=2^-11): {min(ts)*1000:9.2f} ms  maxabs_vs_fp64 {err:.3e}", flush=True)
