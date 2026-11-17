"""Time each ozaki partial matmul at a DiT-realistic shape, fp64_emu (low=fp32) on NPU/GPU."""
import os, sys, time
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import ozaki as O

K, N, T, M = 3072, 12288, 1024, 1

A64 = (np.random.standard_normal((M * T, K)) * 0.5).astype(np.float64)
B64 = (np.random.standard_normal((K, N)) * 0.5).astype(np.float64)
ref = A64 @ B64
A_t = A64.astype(np.float64); B_t = B64.astype(np.float64)
s, u_acc = 6, 2**-24
low_dtype, high_dtype = np.float32, np.float64


def session_for(backend):
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False; so.enable_mem_reuse = False
    ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
    want = ort.OrtHardwareDeviceType.NPU if backend == "htp" else ort.OrtHardwareDeviceType.GPU
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider" and d.device.type == want]
    opts = {"backend_path": qnn_ep.get_qnn_htp_path() if backend == "htp" else qnn_ep.get_qnn_gpu_path(),
            "htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"}
    so.add_provider_for_devices(devs[:1], opts)
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession("_mm_%d_%d_%d_float32_%s.onnx" % (M*T, K, N, backend), so)


for backend in ("htp", "gpu"):
    sess = session_for(backend)
    A_low = [sl.astype(low_dtype) for sl in O.split_matrix(A_t, s, u_acc, K, work_dtype=low_dtype)[0]]
    BT = B_t.T.copy()
    B_low_T = [sl.astype(low_dtype) for sl in O.split_matrix(BT, s, u_acc, K, work_dtype=low_dtype)[0]]
    B_low = [sl.T.copy() for sl in B_low_T]
    print(f"--- backend {backend}: {len(A_low)} A-slices, {len(B_low)} B-slices")
    times = []
    for i in range(s):
        for j in range(s - i):
            t0 = time.perf_counter()
            tmp = sess.run(None, {"A": A_low[i].astype(low_dtype), "B": B_low[j].astype(low_dtype)})[0]
            times.append((time.perf_counter() - t0, i, j))
    times.sort(reverse=True)
    tot = sum(t for t, _, _ in times)
    for t, i, j in times[:8]:
        print(f"   matmul i={i} j={j}: {t*1000:.1f} ms")
    print(f"   total {tot*1000:.1f} ms over {len(times)} matmuls")
