import sys, time, json, warnings, os
import numpy as np
warnings.filterwarnings('ignore')
os.environ["RYZEN_AI_INSTALLATION_PATH"] = r"C:\Program Files\RyzenAI\1.9.0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import ozaki as O
from arch.amd.npu.backend import create_backend

RNG = np.random.default_rng(12345)

def make_matrix(shape, dtype=np.float64):
    return (RNG.standard_normal(shape) * 0.5).astype(dtype)

def time_fn(fn, repeats=2):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        ts.append(t1 - t0)
    return out, float(np.median(ts))

def metrics(C, ref):
    diff = C - ref
    absdiff = np.abs(diff)
    m = {
        'max_abs': float(np.max(absdiff)) if absdiff.size else 0.0,
        'mean_abs': float(np.mean(absdiff)) if absdiff.size else 0.0,
        'rmse': float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        'fro_rel': float(np.linalg.norm(diff,'fro') / (np.linalg.norm(ref,'fro') + 1e-300)),
        'max_rel': float(np.max(absdiff / (np.abs(ref) + 1e-12))) if absdiff.size else 0.0,
    }
    return m

CASES = {
 'small_square': (32,32,32),
 'small_thin': (128,32,8),
 'small_fat': (8,32,128),
 'small_matvec': (128,32,1),
 'large_square': (256,256,256),
 'large_thin': (1024,64,16),
 'large_fat': (16,64,1024),
 'large_matvec': (1024,64,1),
}

SCHEMES = [
 ('fp64_emu', np.float64, np.float32, 2**-24, 6),
 ('fp32_emu', np.float32, np.float16, 2**-11, 8),
]

results = []

for case_name, (m,k,n) in CASES.items():
    print(f"\n=== {case_name}: {m}x{k}x{n} ===", flush=True)
    A64 = make_matrix((m,k), np.float64)
    B64 = make_matrix((k,n), np.float64)
    ref64 = A64 @ B64
    t0=time.perf_counter()
    for _ in range(3): _ = A64 @ B64
    tref64 = (time.perf_counter()-t0)/3.0
    print(f"  ref fp64 median {tref64*1000:.2f} ms", flush=True)
    for scheme_name, high_dt, low_dt, u_acc, s in SCHEMES:
        A_t = A64.astype(high_dt)
        B_t = B64.astype(high_dt)
        ref_t = A_t @ B_t
        t0=time.perf_counter()
        for _ in range(3): _ = A_t @ B_t
        tref_t = (time.perf_counter()-t0)/3.0
        backend = 'npu'
        try:
            sess = create_backend(backend, m, k, n, low_dt)
            def run():
                C, _ = O.ozaki_matmul(A_t, B_t, s, backend, low_dt, high_dt, u_acc, session=sess)
                return C
            C, toz = time_fn(run, repeats=1)
            met = metrics(C.astype(np.float64), ref64.astype(np.float64))
            met_t = metrics(C.astype(np.float64), ref_t.astype(np.float64))
            row = {
                'case': case_name, 'shape': [m,k,n],
                'scheme': scheme_name, 'backend': backend,
                's': s, 'status': 'ok',
                'ozaki_time_ms': toz*1000,
                'ref_fp64_time_ms': tref64*1000,
                'ref_target_time_ms': tref_t*1000,
                'metrics_vs_fp64': met,
                'metrics_vs_target': met_t,
            }
            results.append(row)
            print(f"  {scheme_name} {backend}: ozaki {toz*1000:.2f}ms maxabs_vs_fp64 {met['max_abs']:.3e}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({'case':case_name,'shape':[m,k,n],'scheme':scheme_name,'backend':backend,'s':s,'status':'error','error':repr(e)})
            print(f"  {scheme_name} {backend}: ERROR {e}", flush=True)

with open('results_npu_amd.json','w') as f:
    json.dump(results, f, indent=2)
print("\nSaved results_npu_amd.json")
