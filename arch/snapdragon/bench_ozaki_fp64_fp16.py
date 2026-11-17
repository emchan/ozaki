"""Benchmark ozaki fp64 emulation with fp16 working precision (low_dtype=fp16, high=fp64).

At DiT-realistic shape: A [1,1024,3072] @ B [3072,12288] -> C [1,1024,12288].
Compares: native QNN fp32/fp16 matmul vs ozaki fp64 via fp32 partials vs ozaki fp64 via fp16 partials.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import ozaki as O

K, N, T, M = 3072, 12288, 1024, 1
A64 = (np.random.standard_normal((M * T, K)) * 0.5).astype(np.float64)
B64 = (np.random.standard_normal((K, N)) * 0.5).astype(np.float64)
ref = A64 @ B64
A_t = A64.astype(np.float64); B_t = B64.astype(np.float64)


def plain(kind, dt, repeats=5, warmup=2):
    sess = O.QNNSession(M * T, K, N, dt, kind)
    A = A_t.astype(dt); B = B_t.astype(dt)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); C, _ = O.ozaki_matmul(A, B, 1, kind, dt, np.float64, 2**-24, session=sess); ts.append(time.perf_counter() - t0)
    return min(ts), np.asarray(C)


for backend in ("htp", "gpu"):
    # native single GEMM at low precision
    for dt, dtn in ((np.float32, "fp32"), (np.float16, "fp16")):
        t0, C = plain(backend, dt)
        err = float(np.max(np.abs(np.asarray(C, dtype=np.float64) - ref)))
        flops = 2 * M * T * K * N
        print(f"plain fp64_via_low={dtn} {backend}: {t0*1000:8.2f} ms  maxabs_vs_fp64 {err:.3e}  {flops/(t0)/1e12:5.2f} TFLOPS", flush=True)

    # ozaki fp64 via fp16 partials: low=fp16, high=fp64, s=6
    for s in (4, 6, 8):
        sess = O.QNNSession(M * T, K, N, np.float16, backend)
        t0 = time.perf_counter()
        C, _ = O.ozaki_matmul(A_t, B_t, s, backend, np.float16, np.float64, 2**-24, session=sess)
        dt_ms = (time.perf_counter() - t0) * 1000
        err = float(np.max(np.abs(np.asarray(C, dtype=np.float64) - ref)))
        print(f"ozaki fp64 low=fp16 s={s}: {dt_ms:8.2f} ms  maxabs_vs_fp64 {err:.3e}", flush=True)
