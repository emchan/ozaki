"""Plain fp16 GEMM (DiT shape) on CPU / NPU / Adreno GPU via QNN EP."""
import os, sys, time
import numpy as np
import onnx, onnxruntime as ort
import onnxruntime_qnn as qnn_ep
from onnx import helper, TensorProto, numpy_helper

K, N, T, M = 3072, 12288, 1024, 1
_registered = False


def build(path, ep_dt):
    W = np.random.randn(K, N).astype(np.float16) * 0.02
    W2 = np.random.randn(N, K).astype(np.float16) * 0.02
    inits = [numpy_helper.from_array(W, "W"), numpy_helper.from_array(W2, "W2")]
    nodes = [helper.make_node("MatMul", ["x", "W"], ["a"]),
             helper.make_node("MatMul", ["a", "W2"], ["b"])]
    inp = helper.make_tensor_value_info("x", ep_dt, [M, T, K])
    out = helper.make_tensor_value_info("b", ep_dt, [M, T, K])
    g = helper.make_graph(nodes, "gemm", [inp], [out], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 10
    onnx.save(m, path)
    return os.path.getsize(path)


def run_sess(path, kind, repeats=5, warmup=2):
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False; so.enable_mem_reuse = False
    if kind == "cpu":
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    global _registered
    if not _registered:
        ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
        _registered = True
    want = ort.OrtHardwareDeviceType.NPU if kind == "npu" else ort.OrtHardwareDeviceType.GPU
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider" and d.device.type == want]
    opts = {"backend_path": qnn_ep.get_qnn_htp_path() if kind == "npu" else qnn_ep.get_qnn_gpu_path(),
            "htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"}
    so.add_provider_for_devices(devs[:1], opts)
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(path, so)


def main():
    for kind, label in (("cpu", "CPU"), ("npu", "HTP NPU"), ("gpu", "Adreno GPU")):
        p = f"tmp_gemm_fp16_{kind}.onnx"
        build(p, TensorProto.FLOAT16)
        s = run_sess(p, kind)
        x = np.random.randn(M, T, K).astype(np.float16)
        for _ in range(2): s.run(None, {"x": x})
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); s.run(None, {"x": x}); ts.append(time.perf_counter() - t0)
        ms = min(ts) * 1000
        flops = 4 * T * K * N
        print(f"{label:12s} fp16: {ms:8.2f} ms  {flops/(ms/1000)/1e12:6.2f} TFLOPS", flush=True)


if __name__ == "__main__":
    main()
