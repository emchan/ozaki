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
