import sys, time, numpy as np
sys.path.insert(0, r"C:\path\to\qnn_env\Lib\site-packages")
import onnxruntime as ort
import onnxruntime_qnn as qnn
import onnx
from onnx import helper, TensorProto

def make_session(M,K,N,onnx_dtype,backend):
    A=helper.make_tensor_value_info('A', onnx_dtype, [M,K])
    B=helper.make_tensor_value_info('B', onnx_dtype, [K,N])
    C=helper.make_tensor_value_info('C', onnx_dtype, [M,N])
    node=helper.make_node('MatMul',['A','B'],['C'])
    graph=helper.make_graph([node],'g',[A,B],[C])
    model=helper.make_model(graph, producer_name='test', opset_imports=[helper.make_opsetid('',13)])
    model.ir_version=9
    onnx.save(model, f'tmp_{str(onnx_dtype)}_{backend}.onnx')
    path=getattr(qnn, f'get_qnn_{backend}_path')()
    opts = {'backend_path': path}
    if backend == 'htp':
        opts['htp_performance_mode'] = 'burst'
        opts['enable_htp_fp16_precision'] = '1'   # w8a16: int8 weights x fp16 activations
    return ort.InferenceSession(f'tmp_{str(onnx_dtype)}_{backend}.onnx', providers=[('QNNExecutionProvider',{'backend_path':path, **opts})])

def run_gemm(backend, M, K, N):
    sess = make_session(M, K, N, TensorProto.FLOAT16, backend)
    a = np.random.randn(M, K).astype(np.float16)
    b = np.random.randn(K, N).astype(np.float16)
    sess.run(None, {'A': a, 'B': b})
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(None, {'A': a, 'B': b})
    t1 = time.perf_counter()
    ms = (t1 - t0) / 5 * 1000
    gflops = 2 * M * K * N / ((t1 - t0) / 5) / 1e9
    return ms, gflops

cases = [
    ('htp',  64, 64, 64),
    ('htp',  256, 256, 256),
    ('htp',  1, 3072, 12288),   # DiT-realistic: x[1,1024,3072] @ W[3072,12288]
    ('gpu',  64, 64, 64),
    ('gpu',  256, 256, 256),
    ('gpu',  1, 3072, 12288),   # DiT-realistic: x[1,1024,3072] @ W[3072,12288]
]
for backend, M, K, N in cases:
    try:
        ms, gflops = run_gemm(backend, M, K, N)
        print(f"{backend} w8a16 {M}x{K}x{N}: {ms:.2f} ms {gflops:.2f} GFLOPS")
    except Exception as e:
        print(f"{backend} w8a16 {M}x{K}x{N}: FAILED {str(e)[:200]}")
