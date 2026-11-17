import os, sys, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto

root = r"C:\path\to\RyzenAI\1.9.0"
os.environ["RYZEN_AI_INSTALLATION_PATH"] = root
dll = os.path.join(root, "onnxruntime", "bin", "onnxruntime_providers_vitisai.dll")
ort.register_execution_provider_library("VitisAIExecutionProvider", dll)

M,K,N = 64,64,64
A=helper.make_tensor_value_info('A',TensorProto.FLOAT16,[M,K])
B=helper.make_tensor_value_info('B',TensorProto.FLOAT16,[K,N])
C=helper.make_tensor_value_info('C',TensorProto.FLOAT16,[M,N])
node=helper.make_node('MatMul',['A','B'],['C'])
g=helper.make_graph([node],'g',[A,B],[C])
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)])
m.ir_version=9
os.makedirs('/tmp/ozaki',exist_ok=True)
onnx.save(m,'/tmp/ozaki/tmp_npu_fp16.onnx')

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession('/tmp/ozaki/tmp_npu_fp16.onnx', so, providers=['VitisAIExecutionProvider'], provider_options=[{}])
X=np.random.randn(M,K).astype(np.float16); Y=np.random.randn(K,N).astype(np.float16)
out=sess.run(None,{'A':X,'B':Y})[0]
print('fp16 out shape', out.shape, 'diff', np.max(np.abs(out.astype(np.float32)-(X.astype(np.float32)@Y.astype(np.float32)))))
