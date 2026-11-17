import os, sys, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper

root = r"C:\Program Files\RyzenAI\1.9.0"
os.environ["RYZEN_AI_INSTALLATION_PATH"] = root
dll = os.path.join(root, "onnxruntime", "bin", "onnxruntime_providers_vitisai.dll")
ort.register_execution_provider_library("VitisAIExecutionProvider", dll)

M,K,N = 64,64,64
# int8 weight + scale
W = np.random.randn(K,N).astype(np.float32)*0.02
scale = np.array(0.02, dtype=np.float32)
Wq = np.clip(np.round(W/scale), -127, 127).astype(np.int8)
zp = np.array(0, dtype=np.int8)

# model: dequantize int8 weight to fp16, matmul with fp16 input
A=helper.make_tensor_value_info('A',TensorProto.FLOAT16,[M,K])
C=helper.make_tensor_value_info('C',TensorProto.FLOAT16,[M,N])
init_wq = numpy_helper.from_array(Wq, "Wq")
init_scale = numpy_helper.from_array(np.array(scale,dtype=np.float32), "scale")
init_zp = numpy_helper.from_array(zp, "zp")
dq = helper.make_node('DequantizeLinear', ['Wq','scale','zp'], ['Wdq32'], axis=0)
cast = helper.make_node('Cast', ['Wdq32'], ['Wdq'], to=TensorProto.FLOAT16)
matmul = helper.make_node('MatMul', ['A','Wdq'], ['C'])
g = helper.make_graph([dq, cast, matmul], 'g', [A], [C], [init_wq, init_scale, init_zp])
m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
m.ir_version = 9
os.makedirs('/tmp/ozaki',exist_ok=True)
onnx.save(m, '/tmp/ozaki/tmp_w8a16.onnx')

so=ort.SessionOptions(); so.log_severity_level=3
sess=ort.InferenceSession('/tmp/ozaki/tmp_w8a16.onnx',so,providers=['VitisAIExecutionProvider'],provider_options=[{}])
X=np.random.randn(M,K).astype(np.float16)
out=sess.run(None,{'A':X})[0]
print('w8a16 NPU run OK', out.shape)
ref = X.astype(np.float32) @ (Wq.astype(np.float32)*scale)
print('max abs vs ref', np.max(np.abs(out.astype(np.float32)-ref)))
