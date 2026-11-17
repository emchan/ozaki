"""fp64 via fp32 partials with smaller split counts s (double-double style): s=2,3,4 at DiT shape."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import ozaki as O

K, N, T, M = 3072, 12288, 1024, 1
A64 = (np.random.standard_normal((M * T, K)) * 0.5).astype(np.float64)
B64 = (np.random.standard_normal((K, N)) * 0.5).astype(np.float64)
ref = A64 @ B64
A_t = A64.astype(np.float64); B_t = B64.astype(np.float64)


def full_run(backend, low, high, s, u):
    sess = O.QNNSession(M * T, K, N, low, backend)
    ts = []
    for _ in range(2):
        t0 = time.perf_counter()
        C, _ = O.ozaki_matmul(A_t, B_t, s, backend, low, high, u, session=sess)
        ts.append(time.perf_counter() - t0)
    C = np.asarray(C, dtype=np.float64)
    err = float(np.max(np.abs(C - ref)))
    return min(ts), err


for backend in ("htp", "gpu"):
    for s in (2, 3, 4):
        el, err = full_run(backend, np.float32, np.float64, s, 2 ** -24)
        n_partials = s * (s + 1) // 2
        print(f"ozaki fp64 low=fp32 s={s} ({n_partials} partials): {el*1000:9.2f} ms  maxabs_vs_fp64 {err:.3e}", flush=True)
