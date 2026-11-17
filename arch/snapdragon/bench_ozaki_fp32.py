"""fp32 emulation via ozaki fp16 partials (high=fp32, low=fp16, s=8, u=2^-11) at DiT shape,
on NPU and Adreno GPU; compare timing + accuracy vs fp64 ref (as proxy for fp32-accurate ref)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import ozaki as O

K, N, T, M = 3072, 12288, 1024, 1
A64 = (np.random.standard_normal((M * T, K)) * 0.5).astype(np.float64)
B64 = (np.random.standard_normal((K, N)) * 0.5).astype(np.float64)
ref = A64 @ B64
A_t = A64.astype(np.float32); B_t = B64.astype(np.float32)


def run(backend, s=8, u=2 ** -11):
    sess = O.QNNSession(M * T, K, N, np.float16, backend)   # low dtype fp16 for QNN graph
    ts = []
    for _ in range(2):
        t0 = time.perf_counter()
        C, _ = O.ozaki_matmul(A_t, B_t, s, backend, np.float16, np.float32, u, session=sess)
        ts.append(time.perf_counter() - t0)
    C = np.asarray(C, dtype=np.float32)
    err = float(np.max(np.abs(C.astype(np.float64) - ref)))
    return min(ts), err


for backend in ("htp", "gpu"):
    el, err = run(backend)
    print(f"ozaki fp32_emu (low=fp16, s=8, u=2^-11) {backend}: {el*1000:9.2f} ms  maxabs_vs_fp64 {err:.3e}", flush=True)
