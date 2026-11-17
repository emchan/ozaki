import sys, time, json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import ozaki as O
from arch.apple.backend import create_backend

RNG = np.random.default_rng(999)
M,K,N = 1024,3072,12288
A64 = (RNG.standard_normal((M,K))*0.5).astype(np.float64)
B64 = (RNG.standard_normal((K,N))*0.5).astype(np.float64)
ref64 = A64 @ B64
t0=time.perf_counter()
for _ in range(3):
    _ = A64 @ B64
tref64 = (time.perf_counter()-t0)/3.0
print(f"ref fp64: {tref64*1000:.1f} ms", flush=True)

results=[]
# fp64_emu: high=fp64 low=fp32
for backend in ['cpu','gpu','npu']:
    try:
        high_dt=np.float64; low_dt=np.float32; s=6; u=2**-24
        A_t=A64.astype(high_dt); B_t=B64.astype(high_dt)
        sess=create_backend(backend,M,K,N,dtype=low_dt)
        t0=time.perf_counter()
        C,_=O.ozaki_matmul(A_t,B_t,s,backend,low_dt,high_dt,u,session=sess)
        toz=time.perf_counter()-t0
        diff=np.abs(C.astype(np.float64)-ref64)
        print(f"fp64_emu {backend}: {toz*1000:.1f} ms  maxabs {np.max(diff):.3e}  speedup_vs_fp64 {tref64/toz:.3f}", flush=True)
        results.append({'scheme':'fp64_emu','backend':backend,'ms':toz*1000,'maxabs':float(np.max(diff)),'speedup':float(tref64/toz)})
    except Exception as e:
        print(f"fp64_emu {backend}: ERROR {e}", flush=True)

# fp32_emu: high=fp32 low=fp16 (skip cpu? run but may be slow)
for backend in ['gpu','npu']:
    try:
        high_dt=np.float32; low_dt=np.float16; s=8; u=2**-11
        A_t=A64.astype(high_dt); B_t=B64.astype(high_dt)
        ref_t=A_t@B_t
        sess=create_backend(backend,M,K,N,dtype=low_dt)
        t0=time.perf_counter()
        C,_=O.ozaki_matmul(A_t,B_t,s,backend,low_dt,high_dt,u,session=sess)
        toz=time.perf_counter()-t0
        diff=np.abs(C.astype(np.float64)-ref64)
        print(f"fp32_emu {backend}: {toz*1000:.1f} ms  maxabs_vs_fp64 {np.max(diff):.3e}", flush=True)
        results.append({'scheme':'fp32_emu','backend':backend,'ms':toz*1000,'maxabs':float(np.max(diff))})
    except Exception as e:
        print(f"fp32_emu {backend}: ERROR {e}", flush=True)

os.makedirs('arch/apple/results', exist_ok=True)
with open('arch/apple/results/results_dit_mac.json','w') as f:
    json.dump(results,f,indent=2)
print("Saved arch/apple/results/results_dit_mac.json")