# ozaki — floating-point matrix-mult precision emulation on Apple Silicon (M4)

Evaluation of the **Ozaki method** on an **Apple M4** (not the latest Apple silicon;
newer M5-family parts exist as of 2026), covering the three available compute
devices: **CPU**, **GPU** (Metal/MPS) and **NPU** (Apple Neural Engine, via CoreML).

## Exact device specs (this machine)

- **Model:** Mac mini (Mac16,10), Apple M4
- **CPU:** 10 cores (4 Performance + 6 Efficiency)
- **GPU:** 10 cores (Metal 4)
- **Neural Engine:** 16 cores
- **Memory:** 16 GB unified (memory bandwidth ~120 GB/s)
- **OS:** macOS 26 (Darwin 25.6.0)

## Hardware / backends used

| device | backend used | notes |
|---|---|---|
| CPU | PyTorch CPU | Accelerate/BLAS-backed matmul |
| GPU | PyTorch MPS (Metal) | fp16 + fp32 GEMMs |
| NPU | CoreML MIL (Apple Neural Engine) | converted per shape/dtype; fp16 I/O is cast to fp32 at the boundary |

## Raw GEMM throughput (DiT shape `x[1,1024,3072] @ W[3072,12288]`, 0.1545 TFLOP/chain)

| device | fp16 | fp32 |
|---|---|---|
| CPU | not competitive (fp16 matmul very slow) | ~1.7 TFLOPS |
| GPU (MPS) | ~3.8 TFLOPS | ~3.4 TFLOPS |
| NPU (ANE) | ~1.2 TFLOPS | ~3.1 TFLOPS (accumulation behaves like fp16) |

## Ozaki schemes

### `fp64_emu` (fp64 ← fp32 partials, s = 6)
- **Accuracy:** CPU/GPU excellent (maxabs ~1e-14). NPU good on small shapes, but degrades on large shapes (e.g. 256³: ~0.11 maxabs) — NPU fp32 GEMM accumulation is not full fp32.
- **Speed (DiT):** CPU 4.9 s, GPU 3.7 s, NPU 2.7 s vs native fp64 0.22 s → ~0.04–0.08× (slower, but enables fp64 on GPU/NPU which have no native fp64).
- Small-shape timings: CPU 0.3–3.7 ms, GPU 17–75 ms, NPU 1.3–4.5 ms (8 shapes up to 256³).

### `fp32_emu` (fp32 ← fp16 partials, s = 8)
- **Accuracy:** ~fp16-level only (maxabs ~2e-2 @256³ on all backends), not true fp32.
- **Speed (DiT):** GPU 6.1 s, NPU 8.2 s.
- Small-shape timings: CPU ~199 ms, GPU ~52 ms, NPU ~39 ms (256³).

## Verdict

| device | Ozaki usefulness |
|---|---|
| **CPU** | Native fp64 available; Ozaki is slower and unnecessary. |
| **GPU** | Ozaki fp64 works accurately (via fp32 partials) — useful if fp64 is required. fp32 emulation via fp16 is slower and less accurate than native fp32 (~3.4 TFLOPS). |
| **NPU** | Ozaki fp64 runs but NPU fp32 accumulation is suspect (accuracy degrades at scale). fp32 emulation runs but is slow. Native NPU fp32 (~3.1 TFLOPS) is the better fp32 path. |

**Bottom line:** Ozaki's value here is enabling fp64 on the GPU/NPU, not acceleration. For fp32 workloads, native paths (GPU 3.4 TFLOPS, NPU 3.1 TFLOPS, CPU 1.7 TFLOPS) beat ozaki-fp16 splitting, consistent with the repo's findings on Snapdragon.

## Result files

- `results/results_mac.json` — scheme harness (8 shapes × 2 schemes × 3 backends)
- `arch/apple/{cpu,gpu,npu}/backend.py` — backend implementations used for this evaluation
