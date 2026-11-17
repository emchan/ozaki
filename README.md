# ozaki — floating-point matrix-mult precision emulation via the Ozaki method

Evaluation of the **Ozaki method** for emulating high-precision matrix multiplication
(fp32 / fp64) through **lower-precision GEMMs + error-free splitting / recombination**,
across heterogeneous hardware.

**Background / reference:** K. Ozaki, Y. Uchino, T. Imamura, *"Ozaki Scheme II: A GEMM-oriented
emulation of floating-point matrix multiplication using an integer modular technique"*,
arXiv:2504.08009 [cs.MS] (2025; v4, revised Sep 2026; DOI `10.48550/arXiv.2504.08009`;
IJHPCA 2026, DOI `10.1177/10943420261467787`).
Related: arXiv:2611.03158 (adaptive accuracy-driven splitting),
arXiv:2611.13332 (FP64 performance / limits), CERN mixed-precision talk.

## Why Ozaki is needed

Modern compute hardware is overwhelmingly built around **low-precision arithmetic**:

- **Abundant low-precision FLOPs**: tensor cores / vector units are heavily optimized for
  fp16 / bf16 / fp32 inputs, giving TFLOPS-level throughput (e.g. Hexagon NPU fp16 ~5 TFLOPS,
  Adreno fp16 ~2.3 TFLOPS, NVIDIA Tensor cores).
- **Insufficient high-precision FLOPs**: fp64 (and sometimes even strict fp32 accumulation) is
  either absent, extremely slow, or simply not exposed on consumer/integrated silicon.
  Some consumer GPUs barely have any usable fp64 at all.

**Why Ozaki helps**: when only low-precision GEMMs exist, the high-precision result can still
be recovered — via error-free splitting into `s` low-precision partials plus exact recombination
in the target precision. The hardware's plentiful low-precision flops do the heavy lifting; the
high-precision semantics are synthesized.

## Caveats / disclaimer

- The measurements collected in this repository were **not rigorously controlled**
  (ad-hoc shapes, single-backend runs, limited repeats, etc.).
- This repository is for **educational purposes only**.
- If you would like, you can test in a more controlled environment and check the code for
  correctness and performance.

## The core of the Ozaki method

### Splitting (error-free decomposition into `s` low-precision partials)

For each row $i$ of the target-precision matrix (row max $m_i = \max_j |M_{ij}|$, accuracy target $u$):

$$
\beta = \left\lceil \frac{-\log_2 u + \log_2 k}{2} \right\rceil
$$

$$
\sigma_i = 0.75 \cdot 2^{\lceil \log_2 m_i \rceil} \cdot 2^{\beta}
$$

$$
M_{i,0} = (M_{i,0}^{w} + \sigma_i^{w}) - \sigma_i^{w}
$$

$$
R_{i+1} = M_{i,0} - M_{i,0}
$$

- The first equation sets the split exponent `β`; the second defines the error-free split offset `σ_i` (per row) scaled from the row magnitude and `β`; the third applies the Veltkamp split in working precision `w`; the fourth captures the error term into the next residual.

### Recombination (GEMM as sum of partial products)

$$
C = \sum_{i \ge j} A_i \cdot B_j
$$

With `s` splits per operand there are `s(s+1)/2` distinct partial products; the final high-precision accumulation reconstructs the target-precision product.

### Why it works (intuition)

- **Error-free transformation**: `(x + σ) − σ = x` exactly when `σ` ≥ the rounding-error bound, because subtracting nearly-equal quantities is exact in floating point (no new rounding error is introduced). The split therefore captures the discarded low bits *exactly*.
- **Bounded partials**: choosing $\beta = \lceil (-\log_2 u + \log_2 k) / 2 \rceil$ scales $\sigma$ so that each partial's relative error stays within the target accuracy $u$.
- **Cheap low-precision compute**: each partial GEMM runs on the fast low-precision hardware (e.g. NPU fp16 at ~5 TFLOPS), while the (few) partials are summed in high precision on the CPU.

## Scope

The technique under evaluation: can we run **fp32 / fp64 GEMMs** on hardware that provides only
lower-precision (fp16) or native higher-precision (fp32) arithmetic, by decomposing the target
precision into `s` working-precision terms (`split_matrix`), executing low-precision GEMMs, and
recombining partial products in the high precision?

The implementation lives in `common/` with architecture-specific backends registered under
`arch/<vendor>/<device-family>/`.

### Target hardware classes considered

| class | examples | fp64 native? | measured state in this repo |
|---|---|---|---|
| CPU | x86 (AVX-512, SVE), ARM (Oryon / Neoverse, Graviton) | often yes | used as reference + fallback |
| Intel / x86 matrix engines | Intel AMX (Sapphire Rapids, Granite Rapids), AVX-512 + AVX-512BW/FMA | often yes (fp32+) | **not yet evaluated** |
| iGPU / integrated GPU | Adreno, Intel Iris Xe / Xe-MAX (XMX), Apple (M4), AMD Radeon iGPU, ARM Mali | usually no | Adreno partially explored; **Apple M4 evaluated** (see `arch/apple/`) |
| dGPU | NVIDIA (Tensor cores / TF32), AMD (RDNA matrix engines / VCN fp16) | often no | not yet wired |
| NPU | Qualcomm Hexagon, Arm Ethos, Google TPU | no | **primary ozaki backend so far** |

## Project layout

```text
ozaki/
├── common/                     # common code: the ozaki method core
│   └── ozaki.py                #   _onnx_dtype, split_matrix, ozaki_matmul, QNNSession, numpy_matmul
├── arch/
│   ├── snapdragon/
│   │   ├── cpu/backend.py      # integrated CPU (ORT CPU EP)
│   │   ├── gpu/backend.py      # integrated / discrete GPU
│   │   ├── npu/backend.py      # NPU (QNN EP)
│   │   └── snapdragon.md       # device-specific fp32/fp64 emulation evaluation
│   └── apple/
│       └── silicon/
│           ├── backend.py       # CPU (PyTorch) / GPU (MPS) / NPU (CoreML)
│           └── apple-silicon.md # device-specific fp32/fp64 emulation evaluation
├── bench/                       # experiment / benchmark scripts
│   ├── bench_gemm_dtypes.py
│   ├── bench_gemm_fp16.py
│   ├── bench_gemm_small.py
│   ├── bench_int8.py
│   ├── bench_ozaki_fp16_norm.py
│   ├── bench_ozaki_fp32.py
│   ├── bench_ozaki_fp64_fp16.py
│   ├── bench_ozaki_fp64_s.py
│   └── bench_ozaki_parts.py
├── common/run_tests.py          # ozaki scheme harness (fp64_emu / fp32_emu)
├── summarize.py
├── results.json
└── README.md                    # this file
```

## Quickstart

```powershell
# from the project root (ozaki/)
cd ozaki

# scheme harness: fp64_emu (fp64 <- fp32) and fp32_emu (fp32 <- fp16), backends gpu + npu
python common/run_tests.py
python summarize.py
# capability benchmarks (backend names: cpu | gpu | npu)
python bench\bench_gemm_fp16.py
python bench\bench_gemm_dtypes.py
python bench\bench_gemm_small.py

# ozaki-specific experiments
python bench\bench_ozaki_fp64_fp16.py
python bench\bench_ozaki_fp16_norm.py
python bench\bench_ozaki_fp64_s.py
python bench\bench_ozaki_fp32.py
python bench\bench_ozaki_parts.py
```

## Tested matrix shapes

- **Ozaki scheme harness** (`run_tests.py` — `fp64_emu` / `fp32_emu`, backends `gpu` / `htp`) — 8 shapes:
  - `small_square`: (32, 32, 32)
  - `small_thin`: (128, 32, 8)
  - `small_fat`: (8, 32, 128)
  - `small_matvec`: (128, 32, 1)
  - `large_square`: (256, 256, 256)
  - `large_thin`: (1024, 64, 16)
  - `large_fat`: (16, 64, 1024)
  - `large_matvec`: (1024, 64, 1)
- **Large GEMM** (`bench_ozaki_*` / `bench_gemm_*`): K=3072, N=12288, T=1024, M=1 → C[1,1024,12288] = x[1,1024,3072] @ W[3072,12288] (0.1545 TFLOP/chain).
- **Small-shape tests** (`bench_int8.py` / `bench_gemm_small.py`): (64, 64, 64) and (256, 256, 256).

## Summary of results

### Device GEMM capability (measured on a Snapdragon X Elite, target shape
`x[1,1024,3072]@[3072,12288]`, 0.1545 TFLOP/chain)

| backend | fp16 GEMM | fp32 GEMM | fp64 |
|---|---|---|---|
| Hexagon NPU (burst, fp16 prec / w8a16) | **4.96 TFLOPS** (w8a16; fresh single-GEMM run: htp **13.15 GFLOPS** @ (1,3072,12288) / gpu **11.69 GFLOPS**) | **3.73 TFLOPS** | n/a (must split) |
| Adreno X1-85 GPU | **2.26 TFLOPS** | ~3.8 TFLOPS (theoretical) | n/a (must split) |
| CPU (ORT) | ~0.04 TFLOPS | ~0.13–0.67 TFLOPS | **native** |

- **int8**: the **Hexagon NPU** natively supports int8 GEMMs — **w8a8** (int8 × int8) at **13–18 TFLOPS**
  (30–40 % of the **45 TOPS** NPU peak) and **w4a16** (MatMulNBits) at ~4–5 TFLOPS; the **Adreno GPU** and
  **CPU** also carry int8 capability (Adreno: unmeasured via QNN; CPU: ~2 TOPS estimate). **Test result**:
  an int8 GEMM via ONNX `MatMul` through the QNN EP was attempted on both NPU (`htp`) and GPU (`gpu`) —
  both rejected with `INVALID_GRAPH ... Type 'tensor(int8)' ... is invalid`; the NPU's int8 throughput would
  require a different (quantized-kernel) execution path. By contrast, **w8a16** (int8 weights × fp16 activations,
  via the QNN EP flag `enable_htp_fp16_precision=1`) was measured at **4.96 TFLOPS** (fp16 GEMM, target shape); fresh
  single-GEMM run: htp **13.15 GFLOPS** / gpu **11.69 GFLOPS** (large shape).
- **w8a16 as an emulation path**: w8a16 (int8 weights × fp16 activations) uses fp16-level arithmetic, so
  as an fp32-emulation working precision it does not exceed the fp16 path used in `fp32_emu` (s=8, u_acc=2^-11,
  measured ~2^-11 → not true fp32); as an fp64-emulation working precision it maps to the failed fp16-partial
  route (A2), whereas the validated fp64 route uses low=fp32 partials (s=6, ~fp64 accuracy).

### Ozaki schemes vs native / partial-GEMM paths (target shape)

| # | scheme | high → low | s | u_acc | accuracy | time @ target shape | verdict |
|---|---|---|---|---|---|---|---|
| – | native fp32 GEMM (ozaki s=1) | fp32 → fp32 | 1 | – | fp32 ✓ | **41 ms** | ✅ best available fp32 path |
| – | native fp16 GEMM (ozaki s=1) | fp16 → fp16 | 1 | – | fp16 (~1e-3 rel) | **31 ms** | ✅ native fp16 path |
| 1 | `fp64_emu` (implemented) | fp64 → fp32 | 6 | 2⁻²⁴ | ~fp64 ✓ (rel ≤ ~5e-11 @ 256³) | **~20–30 s** | ✅ accurate; ~500–750x behind native fp32 |
| 2 | A2: ozaki fp64 via fp16 partials | fp64 → fp16 | 4/6/8 | 2⁻²⁴ | **NaN ✗** (sigma overflow) | 24/49/78 s | ❌ broken as-is |
| 2b | A2+: ozaki fp64 via fp16 partials (normalized) | fp64 → fp16 | 4/6/8 | 2⁻¹¹ | errors ~76–107 | ~15–60 s/partial | ❌ not viable yet |
| 1 | `fp32_emu` (implemented) | fp32 → fp16 | 8 | 2⁻¹¹ | ~2⁻¹¹ (not fp32) | 0.8–92 s | ⚠️ runs, wrong accuracy; native fp32 (B2) better |

### All experiments (consolidated)

| # | experiment | result |
|---|-----------|---|
| E1 | fp16 GEMM benchmark | CPU 0.04 / NPU 4.96 / GPU 2.26 TFLOPS |
| E2 | fp32/fp16 GEMMs + ozaki schemes | plain fp32 → NPU 3.73 TFLOPS (41 ms); ozaki fp64 → ~20–30 s, err ~1e-9; ozaki fp32 → ~92 s, err ~2⁻¹¹ |
| E3 | per-partial timing | 21 partials · ~150–191 ms each · ~3.1–3.7 s total |
| E4 | scheme harness | fp64_emu ✓ (~fp64); fp32_emu ~2⁻¹¹ (mis-tuned) |
| E5 | fp64 via fp16 partials (raw) | **NaN** → fail |
| E6 | fp64 via fp16 partials (normalized) | errors ~76–107 → fail/design needed |
| E7 | fp64 via fp32, smaller s | s=2: 6.3 s/err 8.5e-2; s=3: 10.5 s/err 1.07e-3; s=4: 15.7 s/err 1.06e-5 |
| E8 | GEMM ceilings | CPU 0.2–0.7; NPU fp16 4.09–5.12; Adreno 0.49–2.10 TFLOPS |
| E9 | **fp32 emulation via ozaki fp16 partials** | NPU **78 s** / GPU **92 s** · maxabs ~80 (rel ~2⁻¹¹); both backends runnable; ~1000x behind native fp16 |

## Accuracy loss

| path | measured error | loss vs target |
|---|---|---|
| fp64 via ozaki, low=fp32, s=6 | max abs ~1e-9 (rel ≤ ~5e-11 @ 256³) | ≈ fp64 (negligible) |
| fp64 via ozaki, low=fp32, s=4 | max abs 1.06e-5 | ~fp32-ish |
| fp32 via ozaki, low=fp16, s=8, u=2⁻¹¹ | rel ~2.4e-4 | ~2⁻¹³ short of fp32 (2⁻²⁴); effectively fp16-level |

## Work description / cross-hardware evaluation

- **Goal**: run large-scale matmuls at fp64 (or fp32) precision on integrated hardware
  (GPU/NPU/CPU) that lack fp64, using the ozaki splitting method.
- **Method**: Veltkamp-style error-free splitting (`split_matrix`) into `s` low-precision partials,
  backend GEMMs (`cpu` / `gpu` / `npu`), recombine in high precision (`fp64` / `fp32`).
- **Experiments**: capability benchmarks (fp16/fp32 GEMMs), scheme harness (`run_tests.py`), fp64-via-fp16
  (NaN → normalized prototype), fp64-via-fp32 with smaller `s`, fp32 emulation via fp16 partials.
- **Findings**: NPU is the only sensible ozaki backend (~3.7–5 TFLOPS); fp64 emulation via fp32 partials is
  the only validated accurate path (~20–30 s/matmul @ target shape vs 41 ms native fp32); fp32 emulation via ozaki-fp16
  is runnable but ~1000x slower and only ~fp16-accurate; native fp32 GEMMs (NPU: 3.73 TFLOPS / 41 ms) beat
  ozaki-fp16-splitting for fp32.

## Status / next steps

1. fp16-exact ozaki split design for fp64-via-fp16 (A2+ prototype failed).
2. Double-double fp64-via-fp32 (s≈2–4) — accuracy/speed trade not favourable vs s=6.
3. B3: validate NPU fp16 GEMM accumulation precision (single fp16 GEMM → fp32).
4. Native fp32-first adoption for fp32 workloads; ozaki only for fp64.
5. Extend evaluation to additional hardware classes: Intel AMX / AVX-512 matrix engines,
   other iGPUs (Intel Xe-MAX XMX, Mali), discrete AMD/NVIDIA GPUs (fp16/bf16 fp64
   emulation), and other NPUs (Arm Ethos, Google TPU, Intel NPU / Meteor Lake).
   (Apple Silicon M4 now evaluated — see `arch/apple/`.)


## Aggregated results (from `run_ozaki_tests.py`)

```
Speed (ms avg) / Error (max abs avg)
======================================================================================================================================================================
device        vendor      native      target (emu)             cpu                   gpu                   npu            best speedup vs CPU   product               
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
amd           AMD         fp16        fp32              6.3 ms / 2.71e-03     4.2 ms / 2.71e-03     4.8 ms / 2.71e-03         1.51x (gpu)       AMD Ryzen 7 7840HS    
amd           AMD         fp32        fp64              3.0 ms / 5.61e-15     3.1 ms / 5.61e-15     3.2 ms / 5.61e-15         0.98x (gpu)       AMD Ryzen 7 7840HS    
amd           AMD         w8a16       fp16                     n/a                   n/a                   n/a                    n/a           AMD Ryzen 7 7840HS    
apple         Apple       fp16        fp32              31.2 ms / 2.71e-03    35.7 ms / 2.71e-03    12.4 ms / 2.71e-03        2.52x (npu)       Apple M4              
apple         Apple       fp32        fp64              1.0 ms / 6.38e-15     21.1 ms / 6.38e-15    2.0 ms / 1.40e-02         0.51x (npu)       Apple M4              
apple         Apple       w8a16       fp16                     n/a                   n/a            0.2 ms / 2.34e-01             n/a           Apple M4              
snapdragon    Qualcomm    fp16        fp32                     n/a                   n/a                   n/a                    n/a           Snapdragon X1P64100   
snapdragon    Qualcomm    fp32        fp64                     n/a                   n/a                   n/a                    n/a           Snapdragon X1P64100   
snapdragon    Qualcomm    w8a16       fp16                     n/a                   n/a            0.1 ms / 1.83e-01             n/a           Snapdragon X1P64100   
======================================================================================================================================================================

========================================================================
CPU native GEMM capability (reference)
========================================================================
precision   time (ms)     TFLOPS    cpu / product                 
------------------------------------------------------------------------
fp16        54.26         0.001     aarch64                       
fp32        4.35          0.008     aarch64                       
fp64        2.25          0.015     aarch64                       
========================================================================
```

## CPU native GEMM capability (reference)

```
CPU native GEMM capability (reference)
========================================================================
precision   time (ms)     TFLOPS    cpu / product                 
------------------------------------------------------------------------
fp16        50.83         0.001     aarch64                       
fp32        5.39          0.006     aarch64                       
fp64        7.76          0.004     aarch64                       
========================================================================
```
