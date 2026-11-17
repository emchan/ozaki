"""Benchmark realistic DiT-sized GEMMs (fp16 / fp32) on CPU / NPU (HTP) / Adreno GPU (QNN EP),
plus the ozaki fp64_emu (fp32 partials) and fp32_emu (fp16 partials) paths at the same shapes.
"""
import os, sys, time
import numpy as np
import onnx, onnxruntime as ort
import onnxruntime_qnn as qnn_ep
from onnx import helper, TensorProto, numpy_helper

K, N, T, M = 3072, 12288, 1024, 1   # DiT linear-layer shape: C[1,T,N] = x[1,T,K] @ W[K,N]


def build(path, ep_dt):
    W = (np.random.randn(K, N).astype({TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}[ep_dt]) * 0.02)
    W2 = (np.random.randn(N, K).astype({TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}[ep_dt]) * 0.02)
    inits = [numpy_helper.from_array(W, "W"), numpy_helper.from_array(W2, "W2")]
    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["a"]),
        helper.make_node("MatMul", ["a", "W2"], ["b"]),
    ]
    inp = helper.make_tensor_value_info("x", ep_dt, [M, T, K])
    out = helper.make_tensor_value_info("b", ep_dt, [M, T, K])
    g = helper.make_graph(nodes, "gemm", [inp], [out], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.save(m, path)
    return os.path.getsize(path)


def make_sess(path, kind):
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False; so.enable_mem_reuse = False
    if kind == "cpu":
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
    want = ort.OrtHardwareDeviceType.NPU if kind == "npu" else ort.OrtHardwareDeviceType.GPU
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider" and d.device.type == want]
    if not devs:
        raise RuntimeError(f"no QNN device for {kind}")
    opts = {"backend_path": qnn_ep.get_qnn_htp_path() if kind == "npu" else qnn_ep.get_qnn_gpu_path(),
            "htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"}
    so.add_provider_for_devices(devs[:1], opts)
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(path, so)


def run_sess(path, kind, dt, ep_dt, repeats=5, warmup=2):
    s = make_sess(path, kind)
    x = np.random.randn(M, T, K).astype(dt)
    for _ in range(warmup):
        s.run(None, {"x": x})
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); s.run(None, {"x": x}); ts.append(time.perf_counter() - t0)
    ms = min(ts) * 1000
    flops = 4 * T * K * N   # 2 chained GEMMs, 2 flops each
    return ms, flops / (ms / 1000) / 1e12


def main():
    for kind, label in (("cpu", "CPU"), ("npu", "HTP NPU"), ("gpu", "Adreno GPU")):
        for dt, dtn, ep_dt in ((np.float32, "fp32", TensorProto.FLOAT), (np.float16, "fp16", TensorProto.FLOAT16)):
            p = f"tmp_gemm_{kind}_{dtn}.onnx"
            sz = build(p, ep_dt)
            try:
                ms, tf = run_sess(p, kind, dt, ep_dt)
                print(f"{label:12s} {dtn}: {ms:8.2f} ms  {tf:6.2f} TFLOPS", flush=True)
            except Exception as e:
                print(f"{label:12s} {dtn}: FAILED {str(e)[:150]}", flush=True)

    # ---- ozaki paths at this shape: A [M*T, K] @ B [K, N] -> C [M*T, N]
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from common import ozaki as O
    A64 = (np.random.standard_normal((M * T, K)) * 0.5).astype(np.float64)
    B64 = (np.random.standard_normal((K, N)) * 0.5).astype(np.float64)
    ref = A64 @ B64

    for backend in ("htp", "gpu"):
        for scheme_name, high_dt, low_dt, u_acc, s in (
            ("fp64_emu", np.float64, np.float32, 2**-24, 6),
            ("fp32_emu", np.float32, np.float16, 2**-11, 8),
        ):
            A_t = A64.astype(high_dt); B_t = B64.astype(high_dt)
            sess = O.QNNSession(M * T, K, N, low_dt, backend)

            def run_ozaki():
                C, _ = O.ozaki_matmul(A_t, B_t, s, backend, low_dt, high_dt, u_acc, session=sess)
                return np.asarray(C, dtype=np.float64)

            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); C = run_ozaki(); ts.append(time.perf_counter() - t0)
            C = np.asarray(C, dtype=np.float64)
            diff = C - ref
            print(f"ozaki {scheme_name} {backend}: {min(ts)*1000:8.2f} ms  maxabs_vs_fp64 {np.max(np.abs(diff)):.3e}", flush=True)


if __name__ == "__main__":
    main()
