import sys, time, os
import numpy as np
try:
    sys.path.insert(0, r"C:\path\to\qnn_env\Lib\site-packages")
    import onnxruntime as ort
    import onnxruntime_qnn as qnn
except Exception:
    import onnxruntime as ort
    qnn = None
try:
    import onnx
    from onnx import helper, TensorProto
except Exception:
    onnx = None
    helper = None
    TensorProto = None

QNN_LIB = r"C:\path\to\qnn_env\Lib\site-packages"

def _onnx_dtype(npdt):
    if npdt == np.float32:
        return TensorProto.FLOAT
    if npdt == np.float16:
        return TensorProto.FLOAT16
    if npdt == np.float64:
        return None
    raise ValueError(npdt)

class QNNSession:
    def __init__(self, M, K, N, npdt, backend):
        if onnx is None or helper is None or TensorProto is None:
            raise RuntimeError("onnx is required to build a QNNSession")
        self.M, self.K, self.N = M, K, N
        self.npdt = npdt
        self.backend = backend
        onnx_dt = _onnx_dtype(npdt)
        A = helper.make_tensor_value_info('A', onnx_dt, [M, K])
        B = helper.make_tensor_value_info('B', onnx_dt, [K, N])
        C = helper.make_tensor_value_info('C', onnx_dt, [M, N])
        node = helper.make_node('MatMul', ['A', 'B'], ['C'])
        graph = helper.make_graph([node], 'g', [A, B], [C])
        model = helper.make_model(graph, producer_name='ozaki', opset_imports=[helper.make_opsetid('', 13)])
        model.ir_version = 9
        os.makedirs("/tmp/ozaki", exist_ok=True)
        fname = os.path.join("/tmp/ozaki", f"_mm_{M}_{K}_{N}_{np.dtype(npdt).name}_{backend}.onnx")
        onnx.save(model, fname)
        path = getattr(qnn, f'get_qnn_{backend}_path')()
        self.sess = ort.InferenceSession(fname, providers=[('QNNExecutionProvider', {'backend_path': path})])
    def run(self, A, B):
        out = self.sess.run(None, {'A': A, 'B': B})[0]
        return np.array(out, dtype=self.npdt)

def split_matrix(M, s, u, k, work_dtype=np.float32):
    # M: array (m, k) target precision; returns list of s arrays same shape (float64)
    M64 = np.array(M, dtype=np.float64)
    R = M64.copy()
    slices = []
    beta = int(np.ceil((-np.log2(u) + np.log2(k)) / 2.0))
    for p in range(s - 1):
        row_max = np.max(np.abs(R), axis=1)
        # use abs in target precision
        rr = np.where(row_max > 0, row_max, 1.0)
        log2m = np.where(row_max > 0, np.ceil(np.log2(rr)), 0.0)
        sigma = np.where(row_max > 0, 0.75 * (2.0 ** log2m) * (2.0 ** beta), 0.0)
        sig = sigma[:, None]
        # perform Veltkamp rounding in working precision
        Rw = R.astype(work_dtype)
        sigw = sig.astype(work_dtype)
        Mp = ((Rw + sigw) - sigw).astype(np.float64)
        slices.append(Mp)
        R = R - Mp
    slices.append(R)
    return slices, beta

def ozaki_matmul(A, B, s, backend, low_dtype, high_dtype, u_acc, session=None):
    # A: (m,k) high_dtype or castable; B: (k,n)
    A = np.array(A, dtype=high_dtype)
    B = np.array(B, dtype=high_dtype)
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    # split A
    A_slices, betaA = split_matrix(A, s, u_acc, k, work_dtype=low_dtype)
    # split B^T (n x k)
    BT = B.T.copy()
    B_slices_T, betaB = split_matrix(BT, s, u_acc, k, work_dtype=low_dtype)
    B_slices = [sl.T.copy() for sl in B_slices_T]  # each (k,n)
    # cast slices to low dtype
    A_low = [sl.astype(low_dtype) for sl in A_slices]
    B_low = [sl.astype(low_dtype) for sl in B_slices]
    if session is None:
        session = QNNSession(m, k, n, low_dtype, backend)
    C = np.zeros((m, n), dtype=high_dtype)
    # accumulate in high precision
    for i in range(s):
        for j in range(s - i):
            tmp = session.run(A_low[i].astype(low_dtype), B_low[j].astype(low_dtype))
            C = C + tmp.astype(high_dtype)
    return C, session

def numpy_matmul(A, B, dtype):
    return (np.array(A, dtype=dtype) @ np.array(B, dtype=dtype))

if __name__ == '__main__':
    # smoke test
    rng = np.random.default_rng(0)
    m,k,n = 32,32,32
    A = rng.standard_normal((m,k))
    B = rng.standard_normal((k,n))
    for s in [4,6]:
        C, _ = ozaki_matmul(A, B, s, 'htp', np.float32, np.float64, 2**-24)
        ref = A @ B
        print('s', s, 'maxabs', np.max(np.abs(C-ref)))
