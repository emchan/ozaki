import os, sys, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import quantize_dynamic, QuantType

root = r"C:\Program Files\RyzenAI\1.9.0"
os.environ["RYZEN_AI_INSTALLATION_PATH"] = root
dll = os.path.join(root, "onnxruntime", "bin", "onnxruntime_providers_vitisai.dll")
ort.register_execution_provider_library("VitisAIExecutionProvider", dll)

M,K,N = 64,64,64
W = np.random.randn(K,N).astype(np.float32)*0.02
# fp32 model with B as initializer
A=helper.make_tensor_value_info('A',TensorProto.FLOAT,[M,K])
C=helper.make_tensor_value_info('C',TensorProto.FLOAT,[M,N])
init = numpy_helper.from_array(W, "B")
node=helper.make_node('MatMul',['A','B'],['C'])
g=helper.make_graph([node],'g',[A],[C],[init])
m=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)])
m.ir_version=9
onnx.save(m,'tmp_fp32.onnx')

quantize_dynamic('tmp_fp32.onnx','tmp_int8.onnx',weight_type=QuantType.QInt8)
print("quantized")

# run on NPU
so=ort.SessionOptions(); so.log_severity_level=3
sess=ort.InferenceSession('tmp_int8.onnx',so,providers=['VitisAIExecutionProvider'],provider_options=[{}])
X=np.random.randn(M,K).astype(np.float32)
out=sess.run(None,{'A':X})[0]
print('NPU quantized run OK', out.shape)
# CPU reference
ref = X @ W
print('max abs vs ref', np.max(np.abs(out-ref)))
