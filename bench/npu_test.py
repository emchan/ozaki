import os, sys, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto

root = r"C:\path\to\RyzenAI\1.9.0"
os.environ["RYZEN_AI_INSTALLATION_PATH"] = root
sys.path.insert(0, root)

dll = os.path.join(root, "onnxruntime", "bin", "onnxruntime_providers_vitisai.dll")
ort.register_execution_provider_library("VitisAIExecutionProvider", dll)

# find xclbin
xclbin = os.path.join(root, "voe-4.0-win_amd64", "xclbins", "phoenix", "4x4.xclbin")
print("xclbin exists:", os.path.exists(xclbin))

M,K,N = 64,64,64
A=helper.make_tensor_value_info('A',TensorProto.FLOAT,[M,K])
B=helper.make_tensor_value_info('B',TensorProto.FLOAT,[K,N])
C=helper.make_tensor_value_info('C',TensorProto.FLOAT,[M,N])
node=helper.make_node('MatMul',['A','B'],['C'])
g=helper.make_graph([node],'g',[A,B],[C])
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)])
m.ir_version=9
onnx.save(m,'tmp_npu.onnx')

opts = {"xclbin": xclbin}
so = ort.SessionOptions()
so.log_severity_level = 3
sess = ort.InferenceSession('tmp_npu.onnx', so, providers=['VitisAIExecutionProvider'], provider_options=[opts])
X=np.random.randn(M,K).astype(np.float32); Y=np.random.randn(K,N).astype(np.float32)
out=sess.run(None,{'A':X,'B':Y})[0]
print('out shape', out.shape, 'ok')
print('diff', np.max(np.abs(out - (X@Y))))
