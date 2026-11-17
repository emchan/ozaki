import os, sys
backend = sys.argv[1] if len(sys.argv)>1 else 'cpu'
if backend=='npu':
    os.environ["RYZEN_AI_INSTALLATION_PATH"]=r"C:\Program Files\RyzenAI\1.9.0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto

if backend=='npu':
    root=os.environ["RYZEN_AI_INSTALLATION_PATH"]
    ort.register_execution_provider_library("VitisAIExecutionProvider", os.path.join(root,"onnxruntime","bin","onnxruntime_providers_vitisai.dll"))
    providers=["VitisAIExecutionProvider"]; opts=[{}]
elif backend=='gpu':
    providers=["DmlExecutionProvider"]; opts=None
else:
    providers=["CPUExecutionProvider"]; opts=None

M,K,N=64,64,64
A=helper.make_tensor_value_info('A',TensorProto.INT8,[M,K])
B=helper.make_tensor_value_info('B',TensorProto.INT8,[K,N])
C=helper.make_tensor_value_info('C',TensorProto.INT8,[M,N])
node=helper.make_node('MatMul',['A','B'],['C'])
g=helper.make_graph([node],'g',[A,B],[C])
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)])
m.ir_version=9
os.makedirs('/tmp/ozaki',exist_ok=True)
onnx.save(m,'/tmp/ozaki/tmp_int8.onnx')
so=ort.SessionOptions(); so.log_severity_level=3
if opts:
    sess=ort.InferenceSession('/tmp/ozaki/tmp_int8.onnx',so,providers=providers,provider_options=opts)
else:
    sess=ort.InferenceSession('/tmp/ozaki/tmp_int8.onnx',so,providers=providers)
X=np.random.randint(-4,4,(M,K),dtype=np.int8); Y=np.random.randint(-4,4,(K,N),dtype=np.int8)
try:
    out=sess.run(None,{'A':X,'B':Y})[0]
    print(f"{backend}: int8 MatMul OK shape {out.shape}")
except Exception as e:
    print(f"{backend}: int8 MatMul FAILED: {str(e)[:200]}")
