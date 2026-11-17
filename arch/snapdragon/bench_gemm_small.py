import sys, time, numpy as np
sys.path.insert(0, r"C:\path\to\qnn_env\Lib\site-packages")
import onnxruntime as ort
import onnxruntime_qnn as qnn
import onnx
from onnx import helper, TensorProto

def make_session(M,K,N,dtype,onnx_dtype,backend):
    A=helper.make_tensor_value_info('A', onnx_dtype, [M,K])
    B=helper.make_tensor_value_info('B', onnx_dtype, [K,N])
    C=helper.make_tensor_value_info('C', onnx_dtype, [M,N])
    node=helper.make_node('MatMul',['A','B'],['C'])
    graph=helper.make_graph([node],'g',[A,B],[C])
    model=helper.make_model(graph, producer_name='test', opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 9
    onnx.save(model, f'tmp_{dtype}_{backend}.onnx')
    path = getattr(qnn, f'get_qnn_{backend}_path')()
    return ort.InferenceSession(f'tmp_{dtype}_{backend}.onnx', providers=[('QNNExecutionProvider',{'backend_path':path})])

for backend in ['gpu','htp']:
    for dtype,npdt,onnxdt in [('float32',np.float32,TensorProto.FLOAT),('float16',np.float16,TensorProto.FLOAT16)]:
        for (M,K,N) in [(64,64,64),(256,256,256)]:
            sess=make_session(M,K,N,dtype,onnxdt,backend)
            a=np.random.randn(M,K).astype(npdt); b=np.random.randn(K,N).astype(npdt)
            sess.run(None,{'A':a,'B':b})
            t0=time.perf_counter()
            for _ in range(5):
                sess.run(None,{'A':a,'B':b})
            t1=time.perf_counter()
            ms=(t1-t0)/5*1000
            gflops=2*M*K*N/( (t1-t0)/5)/1e9
            print(f"{backend} {dtype} {M}x{K}x{N}: {ms:.2f} ms {gflops:.2f} GFLOPS")
