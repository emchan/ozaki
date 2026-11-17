import os, sys, time, json
import numpy as np
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from arch.apple.npu.backend import create_backend

MODEL_DIR = "/tmp/ozaki"
RESULT_FILE = "arch/apple/results/w8a16_results.json"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)


def run_case(M, K, N):
    sess = create_backend('npu', M, K, N, dtype=np.float16)
    X = np.random.randn(M, K).astype(np.float16)
    Y = np.random.randn(K, N).astype(np.float16)
    # warmup
    sess.run(X, Y)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = sess.run(X, Y)
        ts.append(time.perf_counter() - t0)
    toz = float(np.median(ts))
    ref = X.astype(np.float32) @ Y.astype(np.float32)
    maxabs = float(np.max(np.abs(out.astype(np.float32) - ref)))
    print(f"npu w8a16 {M}x{K}x{N}: OK shape {out.shape} {toz*1000:.2f} ms maxabs {maxabs:.3e}")
    return {
        "case": f"{M}x{K}x{N}",
        "shape": [M, K, N],
        "scheme": "w8a16",
        "backend": "npu",
        "s": 1,
        "status": "ok",
        "ozaki_time_ms": toz * 1000,
        "metrics_vs_fp64": {"max_abs": maxabs},
    }


if __name__ == "__main__":
    cases = [(64, 64, 64), (256, 256, 256)]
    results = []
    ok = True
    for M, K, N in cases:
        try:
            results.append(run_case(M, K, N))
        except Exception as e:
            ok = False
            print(f"npu w8a16 {M}x{K}x{N}: FAILED {str(e)[:300]}")
            results.append({"case": f"{M}x{K}x{N}", "shape": [M, K, N], "scheme": "w8a16",
                           "backend": "npu", "status": "error", "error": repr(e)})
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULT_FILE}")
    raise SystemExit(0 if ok else 1)
EOF