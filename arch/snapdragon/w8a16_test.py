import os, sys, time, json, numpy as np
sys.path.insert(0, r"C:\path\to\qnn_env\Lib\site-packages")
import onnxruntime as ort
import onnxruntime_qnn as qnn
import onnx
from onnx import helper, TensorProto

MODEL_DIR = "/tmp/ozaki"
RESULT_FILE = "arch/snapdragon/results/w8a16_results.json"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)


def make_session(M, K, N, backend="htp"):
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [M, K])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT16, [K, N])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT16, [M, N])
    node = helper.make_node("MatMul", ["A", "B"], ["C"])
    g = helper.make_graph([node], "gemm", [A, B], [C])
    m = onnx.helper.make_model(g, opset_imports=[onnx.helper.make_opsetid("", 13)])
    m.ir_version = 9
    path = os.path.join(MODEL_DIR, f"ozaki_w8a16_{backend}_{M}_{K}_{N}.onnx")
    onnx.save(m, path)

    qnn_path = getattr(qnn, f"get_qnn_{backend}_path")()
    opts = {"backend_path": qnn_path}
    if backend == "htp":
        opts["htp_performance_mode"] = "burst"
        opts["enable_htp_fp16_precision"] = "1"  # w8a16: int8 weights x fp16 activations
    providers = [("QNNExecutionProvider", opts)]
    return ort.InferenceSession(path, providers=providers)


def run_case(M, K, N, backend="htp"):
    sess = make_session(M, K, N, backend)
    X = np.random.randn(M, K).astype(np.float16)
    Y = np.random.randn(K, N).astype(np.float16)
    # warmup
    sess.run(None, {"A": X, "B": Y})
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = sess.run(None, {"A": X, "B": Y})[0]
        ts.append(time.perf_counter() - t0)
    toz = float(np.median(ts))
    ref = X.astype(np.float32) @ Y.astype(np.float32)
    maxabs = float(np.max(np.abs(out.astype(np.float32) - ref)))
    print(f"{backend} w8a16 {M}x{K}x{N}: OK shape {out.shape} {toz*1000:.2f} ms maxabs {maxabs:.3e}")
    return {
        "case": f"{M}x{K}x{N}",
        "shape": [M, K, N],
        "scheme": "w8a16",
        "backend": backend,
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
            results.append(run_case(M, K, N, "htp"))
        except Exception as e:
            ok = False
            print(f"htp w8a16 {M}x{K}x{N}: FAILED {str(e)[:300]}")
            results.append({"case": f"{M}x{K}x{N}", "shape": [M, K, N], "scheme": "w8a16",
                           "backend": "htp", "status": "error", "error": repr(e)})
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULT_FILE}")
    raise SystemExit(0 if ok else 1)
