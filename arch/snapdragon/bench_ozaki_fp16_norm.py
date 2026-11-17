"""fp64 via fp16 partials (ozaki), with sigma normalization into fp16 range.

A [M*T, K] @ B [K, N] -> C [M*T, N]; global pow2 normalization; adaptive beta.
Compares accuracy vs fp64 ref and wall time through QNN NPU/GPU fp16 sessions.

beta: split magnitude = 0.75 * 2^log2m * 2^beta; normalized rows keep log2m small, so we cap beta
to keep sigma inside the working precision's range (fp16 max ~65504, fp32 huge).
"""
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


def split_rows(M, s, beta, work_dtype):
    M64 = np.array(M, dtype=np.float64)
    R = M64.copy()
    slices = []
    for p in range(s - 1):
        row_max = np.max(np.abs(R), axis=1)
        rr = np.where(row_max > 0, row_max, 1.0)
        log2m = np.where(row_max > 0, np.ceil(np.log2(rr)), 0.0)
        sigma = np.where(row_max > 0, 0.75 * (2.0 ** log2m) * (2.0 ** beta), 0.0)
        sig = sigma[:, None]
        Rw = R.astype(work_dtype)
        sigw = sig.astype(work_dtype)
        Mp = ((Rw + sigw) - sigw).astype(np.float64)
        slices.append(Mp)
        R = R - Mp
    slices.append(R)
    return slices


def make_sess(backend, dtype):
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.enable_cpu_mem_arena = False; so.enable_mem_pattern = False; so.enable_mem_reuse = False
    ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
    want = ort.OrtHardwareDeviceType.NPU if backend == "htp" else ort.OrtHardwareDeviceType.GPU
    devs = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider" and d.device.type == want]
    opts = {"backend_path": qnn_ep.get_qnn_htp_path() if backend == "htp" else qnn_ep.get_qnn_gpu_path(),
            "htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"}
    so.add_provider_for_devices(devs[:1], opts)
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(f"_mm_norm_{backend}_{np.dtype(dtype).name}.onnx", so)


def run_case(backend, dtype, s, beta, u=2 ** -11):
    sess = O.QNNSession(M * T, K, N, dtype, backend)
    mx = max(float(np.max(np.abs(A64))), float(np.max(np.abs(B64))))
    e = int(np.ceil(np.log2(mx))) if mx > 0 else 0
    sc = 2.0 ** e
    A = A64 / sc
    B = B64 / sc
    A_sl = split_rows(A, s, beta, dtype)
    B_sl = split_rows(B.T.copy(), s, beta, dtype)
    B_sl = [sl.T.copy() for sl in B_sl]
    C = np.zeros((M * T, N), dtype=np.float64)
    n = 0
    el = 0.0
    for i in range(s):
        for j in range(s - i):
            t0 = time.perf_counter()
            tmp = sess.run(A_sl[i].astype(dtype), B_sl[j].astype(dtype))[0]
            el += time.perf_counter() - t0
            n += 1
            C = C + tmp.astype(np.float64) * (sc * sc)
    err = float(np.max(np.abs(C - ref)))
    return el, err, n, sc


for backend in ("htp",):
    for dtype, dtn, u in ((np.float16, "fp16", 2 ** -11),):
        for s in (4, 6, 8):
            for beta in (15, 10, 6, 3):
                el, err, n, sc = run_case(backend, dtype, s, beta, u)
                print(f"{backend} low={dtn} s={s} beta={beta}: last_matmul={el*1000:8.2f}ms n={n} maxabs={err:.3e} sc={sc}", flush=True)
for backend in ("gpu",):
    for dtype, dtn, u in ((np.float16, "fp16", 2 ** -11),):
        for s, beta in ((6, 6), (8, 3)):
            el, err, n, sc = run_case(backend, dtype, s, beta, u)
            print(f"{backend} low={dtn} s={s} beta={beta}: last_matmul={el*1000:8.2f}ms n={n} maxabs={err:.3e} sc={sc}", flush=True)
