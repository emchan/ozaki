import os, sys, time
import numpy as np
os.environ["RYZEN_AI_INSTALLATION_PATH"] = r"C:\Program Files\RyzenAI\1.9.0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch.amd.npu.backend import create_backend

# DiT-ish shape (T,K)x(K,N)
T, K, N = 1024, 3072, 12288
for dtype in [np.float16, np.float32]:
    try:
        sess = create_backend('npu', T, K, N, dtype)
        X = np.random.randn(T, K).astype(dtype) * 0.02
        W = np.random.randn(K, N).astype(dtype) * 0.02
        for _ in range(1): sess.run(X, W)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); sess.run(X, W); ts.append(time.perf_counter()-t0)
        ms = min(ts)*1000
        tflops = 2*T*K*N/min(ts)/1e12
        print(f"npu {str(dtype):10s} DiT: {ms:8.2f} ms  {tflops:6.2f} TFLOPS")
    except Exception as e:
        print(f"npu {dtype} DiT: ERROR {e}")

for (m,k,n) in [(64,64,64),(256,256,256)]:
    for dtype in [np.float16, np.float32]:
        try:
            sess = create_backend('npu', m, k, n, dtype)
            X = np.random.randn(m,k).astype(dtype)
            W = np.random.randn(k,n).astype(dtype)
            for _ in range(2): sess.run(X,W)
            ts=[]
            for _ in range(5):
                t0=time.perf_counter(); sess.run(X,W); ts.append(time.perf_counter()-t0)
            ms=min(ts)*1000
            tflops=2*m*k*n/min(ts)/1e12
            print(f"npu small {m}x{k}x{n} {str(dtype):10s}: {ms:8.3f} ms {tflops:6.3f} TFLOPS")
        except Exception as e:
            print(f"npu small {m}x{k}x{n} {dtype}: ERROR {e}")
