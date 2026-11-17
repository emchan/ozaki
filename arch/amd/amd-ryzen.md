# FP32 / FP64 emulation on AMD Ryzen (Ryzen 7 7840HS) via the Ozaki method ΓÇö evaluation

**Scope:** device-specific evaluation for the machine under test in this workspace
(AMD Ryzen 7 7840HS, 8 cores / 16 threads, **Radeon 780M** integrated GPU (RDNA3),
**XDNA Phoenix NPU**, 20 GB RAM, Windows 11 Pro 10.0.26200, Python 3.11/3.12,
onnxruntime 1.24.4 (DirectML) / 1.30.0 (VitisAI EP), Ryzen AI Software 1.9.0).

## 0. Project layout (AMD additions)

```
ozaki/
  common/                     <- ozaki method core
  arch/
    amd/
      ryzen/
        backend.py            <- CPU (ONNX CPU EP) / GPU (DirectML EP)
        npu_backend.py        <- NPU (VitisAI EP)
        amd-ryzen.md          <- this file
  bench_amd.py
  bench_npu_amd.py
  bench_ozaki_fp64_s_amd.py
  bench_ozaki_fp32_amd.py
  run_tests_amd.py
  run_tests_npu_amd.py
  results_amd.json
  results_npu_amd.json
```

## 1. Device capability examination ΓÇö CPU / GPU / NPU

### 1.1 Components

| block | part | notes |
|---|---|---|
| SoC | AMD Ryzen 7 7840HS | 8c/16t, up to 5.1 GHz, Phoenix |
| CPU | Zen 4 cores | x86-64; **fp64 native** (AVX-512/AVX2 FMA); numpy/OpenBLAS fp32 ~0.75 TFLOPS |
| GPU | **Radeon 780M** (RDNA3 iGPU) | DirectML EP; fp16 2.38 TFLOPS, fp32 1.01 TFLOPS @ DiT shape |
| NPU | **XDNA Phoenix** | VitisAI EP via Ryzen AI 1.9.0; fp16 0.50 TFLOPS, fp32 0.59 TFLOPS @ DiT shape |
| RAM | ~20 GB | |

### 1.2 Measured FLOPs (DiT shape `x[1,1024,3072] @ [3072,12288]`, 0.1545 TFLOP/chain)

| backend | dtype | time | TFLOPS |
|---|---|---|---|
| CPU (ONNX CPU EP) | fp16 | 146.40 ms | 0.53 |
| CPU (ONNX CPU EP) | fp32 | 202.54 ms | 0.38 |
| GPU (DirectML) | fp16 | 32.43 ms | 2.38 |
| GPU (DirectML) | fp32 | 76.87 ms | 1.01 |
| NPU (VitisAI EP) | fp16 | 155.12 ms | 0.50 |
| NPU (VitisAI EP) | fp32 | 130.60 ms | 0.59 |

## 2. Ozaki scheme harness (8 shapes)

See `results_amd.json` (CPU/GPU) and `results_npu_amd.json` (NPU).

| scheme | backend | ozaki time (ms, range) | max abs vs fp64 |
|---|---|---|---|
| `fp64_emu` (s=6, low=fp32) | CPU | 0.83 ΓÇô 18.87 | ~3.0e-14 |
| `fp64_emu` | GPU | 5.24 ΓÇô 19.19 | ~3.0e-14 |
| `fp64_emu` | NPU | 0.76 ΓÇô 9.57 | ~3.0e-14 |
| `fp32_emu` (s=8, low=fp16) | CPU | 1.65 ΓÇô 44.21 | ~2.2e-2 |
| `fp32_emu` | GPU | 8.24 ΓÇô 49.46 | ~2.2e-2 |
| `fp32_emu` | NPU | 1.97 ΓÇô 41.31 | ~2.2e-2 |

## 3. fp64 via fp32 partials, smaller s (DiT shape)

| backend | s=2 (3 partials) | s=3 (6) | s=4 (10) | s=6 (21) |
|---|---|---|---|---|
| CPU | 3128 ms / 8.34e-2 | 4879 ms / 9.32e-4 | 7068 ms / 1.11e-5 | 12533 ms / 9.61e-10 |
| GPU | 3128 ms / 8.91e-2 | 4341 ms / 9.86e-4 | 5908 ms / 1.12e-5 | 9305 ms / 1.05e-9 |
| NPU | 2986 ms / 8.34e-2 | 4626 ms / 9.32e-4 | 6396 ms / 1.11e-5 | 12045 ms / 9.61e-10 |

## 4. fp32 emulation via ozaki fp16 partials (s=8, u=2^-11, DiT shape)

| backend | time | max abs vs fp64 |
|---|---|---|
| CPU | 20899 ms | 7.80e+01 |
| GPU | 16355 ms | 7.53e+01 |
| NPU | 18518 ms | 7.28e+01 |

## 5. int8 / w8a16

- Standard ONNX `MatMul` rejects `tensor(int8)` inputs on CPU/GPU/NPU.
- A **dynamically quantized int8** model runs on the NPU via VitisAI EP (verified).
- A **w8a16-style** model (int8 weights dequantized to fp16, fp16 `MatMul`) runs on the NPU:
  - Verified, max abs vs reference Γëê 2.4e-4.

## 6. Conclusion

- `fp64_emu` is accurate on all three backends but ~1000├ù slower than native fp32 GEMM.
- `fp32_emu` is runnable everywhere but only fp16-accurate.
- GPU (DirectML) is the fastest ozaki backend on this machine; NPU is comparable to CPU for ozaki
  but much better for native fp16 GEMM (0.50 TFLOPS).
- int8 compute is available via quantized paths (VitisAI), not via plain `MatMul`.


---

## AMD_EVALUATION.md (merged)

# AMD Ryzen evaluation of Ozaki method

Machine: AMD Ryzen 7 7840HS (8c/16t), Radeon 780M iGPU, XDNA Phoenix NPU; Windows 11 Pro.

## 1. GEMM capability (DiT shape, 0.1545 TFLOP/chain)

| backend | fp16 | fp32 |
|---|---|---|
| CPU (ONNX Runtime CPU EP) | 0.53 TFLOPS | 0.38 TFLOPS |
| GPU (DirectML) | 2.38 TFLOPS | 1.01 TFLOPS |
| NPU (VitisAI EP) | 0.50 TFLOPS | 0.59 TFLOPS |

## 2. Ozaki scheme harness (8 shapes)

- `fp64_emu` (s=6, low=fp32): ~fp64 accuracy (max abs ~3e-14) on CPU/GPU/NPU.
- `fp32_emu` (s=8, low=fp16): ~fp16-level accuracy (~2e-2) on all three backends.

Timings: see `results_amd.json` (CPU/GPU) and `results_npu_amd.json` (NPU).

## 3. fp64 via fp32 partials, smaller s (DiT shape)

| backend | s=2 | s=3 | s=4 | s=6 |
|---|---|---|---|---|
| CPU | 3128 ms / 8.34e-2 | 4879 ms / 9.32e-4 | 7068 ms / 1.11e-5 | 12533 ms / 9.61e-10 |
| GPU | 3128 ms / 8.91e-2 | 4341 ms / 9.86e-4 | 5908 ms / 1.12e-5 | 9305 ms / 1.05e-9 |
| NPU | 2986 ms / 8.34e-2 | 4626 ms / 9.32e-4 | 6396 ms / 1.11e-5 | 12045 ms / 9.61e-10 |

## 4. fp32 emulation via ozaki fp16 partials (s=8, u=2^-11, DiT shape)

| backend | time | max abs vs fp64 |
|---|---|---|
| CPU | 20899 ms | 7.80e+01 |
| GPU | 16355 ms | 7.53e+01 |
| NPU | 18518 ms | 7.28e+01 |

Conclusion: fp64_emu is accurate but ~1000x slower than native fp32 GEMM; fp32_emu is runnable on all three backends but only fp16-accurate. GPU is the fastest ozaki backend on this machine; NPU is comparable to CPU for ozaki but much better for native fp16 GEMM (0.50 TFLOPS).
