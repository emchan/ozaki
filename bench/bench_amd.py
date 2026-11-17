import sys, time, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def create_backend(backend, M, K, N, dtype=np.float32, model_dir=None):
    if backend == 'cpu':
        from arch.amd.cpu.backend import create_backend as cb
    elif backend == 'gpu':
        from arch.amd.gpu.backend import create_backend as cb
    else:
        raise ValueError(f"unknown backend {backend}")
    return cb(backend, M, K, N, dtype, model_dir=model_dir)

# DiT shape: x[1,1024,3072] @ W[3072,12288]
T, K, N = 1024, 3072, 12288
M = 1

def bench(backend, dtype, repeats=5):
    # We'll GEMM (T, K) x (K, N) -> (T, N) ignoring M
    sess = create_backend(backend, T, K, N, dtype)
    X = np.random.randn(T, K).astype(dtype) * 0.02
    W = np.random.randn(K, N).astype(dtype) * 0.02
    for _ in range(2):
        sess.run(X, W)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sess.run(X, W)
        ts.append(time.perf_counter() - t0)
    ms = min(ts) * 1000
    flops = 2 * T * K * N
    return ms, flops / (min(ts)) / 1e12

for backend in ['cpu', 'gpu']:
    for dtype in [np.float16, np.float32]:
        try:
            ms, tflops = bench(backend, dtype)
            print(f"{backend:4s} {str(dtype):10s}: {ms:8.2f} ms  {tflops:6.2f} TFLOPS")
        except Exception as e:
            print(f"{backend:4s} {str(dtype):10s}: ERROR {e}")
# small shapes
for (m,k,n) in [(64,64,64),(256,256,256)]:
    for backend in ['cpu','gpu']:
        for dtype in [np.float16, np.float32]:
            try:
                sess = create_backend(backend, m, k, n, dtype)
                X = np.random.randn(m,k).astype(dtype)
                W = np.random.randn(k,n).astype(dtype)
                for _ in range(2): sess.run(X,W)
                ts=[]
                for _ in range(5):
                    t0=time.perf_counter(); sess.run(X,W); ts.append(time.perf_counter()-t0)
                ms=min(ts)*1000
                tflops=2*m*k*n/(min(ts))/1e12
                print(f"small {backend:4s} {m}x{k}x{n} {str(dtype):10s}: {ms:8.3f} ms {tflops:6.3f} TFLOPS")
            except Exception as e:
                print(f"small {backend} {m}x{k}x{n} {dtype}: ERROR {e}")
