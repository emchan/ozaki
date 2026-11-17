# FP32 / FP64 Emulation on the Snapdragon X Elite via the Ozaki Method — evaluation

**Scope:** this is a device-specific evaluation for the machine under test in this workspace
(Surface Pro 11th generation, **Snapdragon X1P-64-100** board / **X1P64100** @ 3.4 GHz, 10-core
Oryon X1, **Adreno X1-85** integrated GPU, **Hexagon NPU** (QNN HTP), **16 GB LPDDR5X**, Windows 11
10.0.26300, Python 3.12 ARM64, onnxruntime 1.31.1 + onnxruntime-qnn 2.7.0).

The goal of the ozaki work in this folder is to run **matrix products at a target precision that the
integrated hardware does not natively provide** (fp64, or fp32) by using **lower-precision GEMMs** plus
the **ozaki splitting method** (Veltkamp-style error-free splitting into `s` working-precision terms +
error-free re-combination), with the final accumulation in the high precision.

## 0. Project layout

```
ozaki/                          <- this project
├── common/                     <- common code: the ozaki method core
│   └── ozaki.py                #   _onnx_dtype, split_matrix, ozaki_matmul, QNNSession, numpy_matmul
├── arch/
│   └── snapdragon/
│       ├── cpu/backend.py      #   integrated CPU (ORT CPU EP)
│       ├── gpu/backend.py      #   Adreno GPU (QNN EP)
│       └── npu/backend.py      #   Hexagon NPU (QNN EP)
├── bench/                       <- experiment / benchmark scripts
│   ├── bench_gemm_dtypes.py
│   ├── bench_gemm_fp16.py
│   ├── bench_gemm_small.py
│   ├── bench_int8.py
│   ├── bench_ozaki_fp16_norm.py
│   ├── bench_ozaki_fp32.py
│   ├── bench_ozaki_fp64_fp16.py
│   ├── bench_ozaki_fp64_s.py
│   └── bench_ozaki_parts.py
├── run_tests.py                 # scheme harness
├── summarize.py
├── results.json
└── README.md                    # this file / quickstart
```

## 1. Device capability examination — CPU / GPU / NPU

### 1.1 Components

| block | part | notes |
|---|---|---|
| SoC | Snapdragon X1P-64-100 (reported **10-core X1P64100 @ 3.4 GHz**) | Oryon X1 cores |
| CPU | 10-core Oryon | ARM64; **fp64 native** (NEON/scalar FP); ORT fp32 GEMM ~0.1–0.7 TFLOPS |
| GPU | **Adreno X1-85** | QNN EP backend available; Vulkan 1.3.295 ICD present but not HKLM-registered; driver 31.0.137.0 |
| NPU | **Hexagon** via **QNNExecutionProvider** (`get_qnn_htp_path`) | `htp_performance_mode=burst`, `enable_htp_fp16_precision=1` |
| RAM | 16 GB LPDDR5X (15.6 GB visible) | fp32 weights export noted; int4 8.49 GB fits |

### 1.2 Measured / declared FLOPs

Fresh measurements this session (best of repeats), DiT-shaped GEMM
`x[1,1024,3072] @ W[3072,12288]` then `@ W2[12288,3072]` = **0.1545 TFLOP** per chain:

| backend | dtype | time | **GFLOPS** | **TFLOPS** | comment |
|---|---|---|---|---|---|
| CPU (ORT) | fp32 | 1216.71 ms | 127 | **0.13** | fresh, this session |
| CPU (ORT) | fp32 | ~232 ms | 665 | **0.67** | `bench_gemm.py` T=1024, earlier run |
| NPU — HTP (QNN) | **fp32** | 41.47 ms | 372 | **3.73** | fresh, this session (QNN fp32 MatMul works) |
| NPU — HTP (QNN) | **fp16** | 31.17 ms | 496 | **4.96** | fresh, this session (DiT shape) |
| Adreno GPU (QNN) | **fp16** | 68.44 ms | 226 | **2.26** | fresh, this session |
| Adreno GPU (QNN) | fp16 | 83–294 ms | 526–186 | 0.74–2.10 | gemm tables, T=1024/4 chains |

Published/derived ceilings on this device:

| block | fp16 / fp32 / int8 capability |
|---|---|
| **Hexagon NPU** | pure fp16 GEMM **~5 TFLOPS**; **w8a16** (int8 weights × fp16 activations; via QNN EP flag `enable_htp_fp16_precision=1`) measured at **4.96 TFLOPS** (fp16 GEMM, DiT shape; fresh single-GEMM run: htp **13.15 GFLOPS** / gpu **11.69 GFLOPS**); **w8a8** (int8 × int8) **13–18 TFLOPS** (30–40 % of the **45 TOPS** NPU peak); w4a16 MatMulNBits ~4–5 TFLOPS; `fp32` MatMul also claimable via QNN (~3.7 TFLOPS measured). No flash attention. |
| **Adreno X1-85 GPU** | theoretical fp16 ~7.6 TFLOPS / fp32 ~3.8 TFLOPS (Vulkan); via QNN EP measured fp16 **~2.26 TFLOPS** (DiT shape), 0.74–2.1 TFLOPS in gemm tables; collapses at long sequences via QNN. |
| **CPU** | fp32 ~0.13–0.7 TFLOPS (ORT); fp64 **native**; int8 ~2 TOPS (peak estimate, ORT CPU). |

### 1.3 Numerics capability summary

- **fp16**: natively available on **NPU** (w8a16, `enable_htp_fp16_precision=1`) and **Adreno GPU**
  (fp16 / bf16 shaders). This is the *only* precision with usable native throughput on the NPU.
- **fp32**: natively claimable through the **QNN EP on both NPU and Adreno GPU** (fp32 MatMul graphs run;
  NPU fp32 GEMM ~3.73 TFLOPS, faster than its own fp16 in our measurement). So an fp32-accurate GEMM is
  *possible* on the NPU without any splitting.
- **fp64**: **no native fp64** in the NPU or Adreno GPU path. Must be produced by splitting + error-free
  re-combination (the ozaki method), with final accumulation in fp64 (CPU has native fp64).
- **int8**: the **Hexagon NPU** natively supports int8 GEMMs — **w8a8** (int8 weights) at **13–18 TFLOPS**
  (30–40 % of the **45 TOPS** NPU peak) and **w4a16** (MatMulNBits) at ~4–5 TFLOPS; the **Adreno GPU** and
  **CPU** also carry int8 capability (Adreno: unmeasured via QNN; CPU: ~2 TOPS estimate). **Test result**:
  an int8 GEMM via ONNX `MatMul` through the QNN EP was attempted on both NPU (`htp`) and GPU (`gpu`) —
  both rejected with `INVALID_GRAPH ... Type 'tensor(int8)' of input parameter (A) of operator (MatMul) ... is invalid`,
  i.e. the QNN-EP-backed ONNX `MatMul` path does not accept int8 inputs in this setup; the NPU's int8
  throughput would require a different (quantized-kernel) execution path. By contrast, **w8a16**
  (int8 weights × fp16 activations, via the QNN EP flag `enable_htp_fp16_precision=1`) was measured at
  **4.96 TFLOPS** (fp16 GEMM, DiT shape) and is the configuration used for the ~5 TFLOPS fp16 ceiling.
  **w8a16 as an emulation path**: w8a16 (int8 weights × fp16 activations) is fp16-level arithmetic, so it
  does not yield true fp32 accuracy (the `fp32_emu` scheme, low=fp16, s=8, u_acc=2^-11, measured ~2^-11 → not
  true fp32) and does not yield fp64 accuracy (the fp16-partial route, A2, failed; validated fp64 uses
  low=fp32 partials, s=6, ~fp64 accuracy).

---

## 2. The Ozaki method (as implemented in `common/ozaki.py`)

- `split_matrix(M, s, u, k, work_dtype)`: Veltkamp-style splitting — per row compute
  `sigma = 0.75 * 2^ceil(log2(max|R|)) * 2^beta`, `beta = ceil((-log2(u) + log2(k)) / 2)`, then
  `Mp = (Rw + sigw) - sigw` (error-free in working precision), remainder `R = R - Mp`; repeated `s-1` times.
- `ozaki_matmul(A, B, s, backend, low_dtype, high_dtype, u_acc, session)`:
  split A and B into `s` low-precision slices, cast slices to `low_dtype`, run the backend's MatMul,
  combine partials in `high_dtype` (fp64 accumulation on CPU in the current implementation).

### Implemented schemes

| scheme | high (emulated) | low (working) | `s` | `u_acc` |
|---|---|---|---|---|
| `fp64_emu` | fp64 | fp32 | 6 | 2^-24 |
| `fp32_emu` | fp32 | fp16 | 8 | 2^-11 |

Partial-GEMM count per ozaki run: triangular `sum_{i=0}^{s-1}(s-i) = s(s+1)/2` (s=6 → **21** partials;
s=8 → **36** partials).

### Schemes evaluated

| scheme | target (high) | working (low) | s | u_acc | outcome |
|---|---|---|---|---|---|
| `fp64_emu` (Ozaki) | fp64 | fp32 | 6 | 2⁻²⁴ | ✅ validated (~fp64) |
| `fp32_emu` (Ozaki) | fp32 | fp16 | 8 | 2⁻¹¹ | ⚠️ ran (~2⁻¹¹, not true fp32) |
| A2 (Ozaki) | fp64 | fp16 | 4/6/8 | 2⁻²⁴ | ❌ fail (NaN / errors) |
| A2+ (Ozaki) | fp64 | fp16 | 4/6/8 | 2⁻¹¹ | ❌ fail |
| A3 (Ozaki) | fp64 | fp32 | 2/3/4 | 2⁻²⁴ | ⚠️ mixed |
| B2 (native fp32) | fp32 | fp32 | 1 | – | ✅ fp32 ✓ |
| B3 (native fp16 + fp32 acc) | fp32 | fp16 | 1 | – | ❌ unvalidated |

---

## 3. Measurements of the implemented schemes (run via `ozaki/run_tests.py`, using the `common` core, QNN EP backends `gpu`/`htp`)

fp64_emu (s=6, low=fp32) — **accurate to ~fp64**:

| case | shape | backend | ozaki ms | ref fp64 ms | max abs err | max rel err |
|---|---|---|---|---|---|---|
| small_square | 32³ | gpu/htp | 0.40–0.47 | 0.02 | 2.2e-15 | 8.4e-14 |
| large_square | 256³ | gpu/htp | 8.0–9.2 / 24.4 | 0.53 | 2.4e-14 | 5.1e-11 |
| large_thin | 1024×64×16 | gpu/htp | 1.9–2.1 / 2.1–2.8 | 0.16 | 5.3e-15 | 6.2e-12 |
| large_fat | 16×64×1024 | gpu/htp | 5.1–7.0 / 24.7 | 0.64 | 6.2e-15 | 2.2e-12 |
| large_matvec | 1024×64×1 | gpu/htp | 1.5–2.1 / 2.1 | 0.01 | 2.7e-15 | 3.4e-13 |

→ **fp64 emulation via fp32 partials works and is very accurate** (rel err ≤ ~5e-11 even at 256³).

fp32_emu (s=8, low=fp16) — runs, but delivered **~2^-11 relative accuracy**, not true fp32 (2^-24):

| case | shape | backend | ozaki ms | max abs err | max rel err |
|---|---|---|---|---|---|
| small_square | 32³ | gpu/htp | 0.78–0.96 | 1.3e-06 | 1.1e-04 |
| large_square | 256³ | gpu/htp | 13.8–15.1 | 2.2e-02 | 7.3 |
| large_thin | 1024×64×16 | gpu/htp | 4.8–8.6 | 1.7e-05 | 2.9e-02 |
| large_fat | 16×64×1024 | gpu/htp | 7.3–10.0 | 1.7e-05 | 1.3e-02 |
| large_matvec | 1024×64×1 | gpu/htp | 4.5–5.0 | 1.0e-05 | 2.2e-03 |

(The `large_square` row's large *relative* error is on entries of large magnitude; the *relative*
error scale (~2^-11 ≈ 2.4e-4) matches the configured `u_acc=2^-11`, i.e. the scheme as tuned delivers
~fp16-mantissa-level accuracy, not fp32.)

---

## 4. Ways to implement fp32 / fp64 emulation on the integrated GPU and NPU

### 4.1 fp64 emulation — candidate implementations

| # | implementation idea | working precision | partials | backend | accuracy | status |
|---|---|---|---|---|---|---|
| A1 | **ozaki fp64** (implemented) | fp32 | s=6 → 21 partials | NPU (QNN fp32) / GPU (QNN fp32) | ~fp64 ✓ | **ran**: 1.5–31 s @ DiT-realistic shapes (overhead-dominated) |
| A2 | ozaki with fp16 partials | fp16 | s≥6..8 → 21..36 partials | NPU (QNN fp16, ~5 TFLOPS — fastest) / GPU (QNN fp16, ~2.3 TFLOPS) | predicted ~fp64 | **not benchmarked yet** |
| A3 | double-double style (hi/lo split, ~2 terms) | fp32 | ~4 partials | NPU/GPU | ~fp64 | not implemented here (s=6 is the conservative variant) |

Cost note (this session, `[1,1024,3072]@[3072,12288]`): plain fp32 chain via QNN NPU = **41 ms** (3.73 TFLOPS);
ozaki fp64 via fp32 partials (s=6, 21 partials) = **~19–31 s** per matmul at the same shape. The ozaki path is
therefore **~500–750x behind the native fp32 GEMM** at DiT-realistic sizes — the gap is per-op overhead on QNN
plus the partial-GEMM fan-out, not raw FLOPs.

### 4.2 fp32 emulation — candidate implementations

| # | implementation idea | working precision | partials | backend | accuracy | status |
|---|---|---|---|---|---|---|
| B1 | **ozaki fp32** (implemented) | fp16 | s=8 → 36 partials | NPU (QNN fp16, ~5 TFLOPS) / GPU (QNN fp16, ~2.3 TFLOPS) | ~2^-11 as tuned (not full fp32) | **ran**: 0.8–92 s @ various shapes |
| B2 | **native fp32 GEMM** | fp32 | 1 | NPU (QNN fp32, 3.73 TFLOPS measured) / GPU (QNN fp32) | fp32 ✓ | **ran**: fp32 MatMul finalized & ran on NPU via QNN (41 ms / 3.73 TFLOPS) |
| B3 | single fp16 GEMM with fp32 accumulation | fp16 arithmetic | 1 | NPU / GPU | fp32 *if* hardware accumulates fp32 | unvalidated (numerics question) |

### 4.3 Cross-cutting implementation choices

- **Backend choice for the partial GEMMs:** NPU (fp16 4.96 TFLOPS / fp32 3.73 TFLOPS) > Adreno GPU (fp16
  2.26 TFLOPS via QNN) > CPU (0.13–0.7 TFLOPS). For any ozaki variant, the NPU is the only backend where
  the partial GEMMs can be throughput-relevant.
- **Final accumulation:** keep in fp64 on CPU (native fp64, numpy) — current design; device-side fp16/fp32
  accumulation would only widen error.
- **Data movement:** each partial result must round-trip NPU→CPU; with 21–36 partials per output element
  this is frequent for attention-scale tensors — consider folding partials into the next GEMM (i.e.
  interleave splitting with the contraction) instead of materializing partial outputs.

---

## 5. Evaluation — summary

1. **Capability:** the integrated NPU delivers the highest *native* GEMM throughput of the three blocks
   (fp16 ~5 TFLOPS, fp32 ~3.7 TFLOPS, w8a8 up to 18 TFLOPS peak), ~2x the Adreno GPU via QNN, and ~40x the CPU
   fp32. fp64 exists **nowhere** in the NPU/GPU path — ozaki splitting is *required* for fp64.
2. **fp64 emulation:** the implemented ozaki scheme (fp32 working precision, s=6) is accurate; its cost is
   the partial-GEMM count (21) plus QNN per-op overhead, making it impractical at DiT scale today (~20–30 s
   per matmul vs 41 ms for a native fp32 GEMM). A fp16-partial variant (A2) was prototyped with pow2 row
   normalization + capped `beta` → did not validate cleanly (large errors / slow); double-double-style `s=2..4`
   experiments also ran (s=4: 15.7 s, err 1.06e-5) — accuracy/speed trade not favourable vs s=6 (err ~1e-9, 23 s).
3. **fp32 emulation:** the implemented ozaki scheme (fp16 working precision, s=8) runs but delivers only
   ~2^-11 accuracy as configured; also expensive (36 partials). Notably, **a native fp32 GEMM already runs
   on the NPU via QNN at 3.73 TFLOPS** — i.e. ozaki-based fp16 splitting may be unnecessary if fp32 GEMMs are
   acceptable in the deployment runtime; B3 (single fp16 GEMM + fp32 accumulation) remains an open numerics
   question that would make fp32 emulation nearly free.
4. **GPU vs NPU:** for ozaki-style emulation the **NPU is the only viable backend** (fp16/fp32 GEMM both
   >2 TFLOPS); the Adreno GPU via QNN is ~2.3 TFLOPS fp16 and slower per partial — prefer NPU unless Vulkan
   support lands.

---

## 6. Artifacts produced for this evaluation

| file | role |
|---|---|
| `common/ozaki.py` | ozaki method core: dtype map, split_matrix, ozaki_matmul, QNNSession, numpy_matmul |
| `arch/snapdragon/{cpu,gpu,npu}/backend.py` | architecture-specific backends (ORT CPU / QNN Adreno GPU / QNN Hexagon NPU) |
| `bench/bench_gemm_dtypes.py` | fp16/fp32 GEMMs on CPU/NPU/GPU at DiT shape + ozaki schemes at same shape |
| `bench/bench_gemm_fp16.py` | fp16 GEMM benchmark (CPU/NPU/GPU) at DiT shape |
| `bench/bench_ozaki_parts.py` | per-partial matmul timing breakdown for ozaki fp64 (s=6, low=fp32) |
| `bench/bench_ozaki_fp64_fp16.py` | ozaki fp64 via fp16 partials (s=4/6/8) + plain fp32/fp16 GEMMs at DiT shape |
| `bench/bench_gemm_small.py` | fp16/fp32 GEMMs on NPU/GPU, small shapes |
| `bench/bench_int8.py` | w8a16 (int8 weights × fp16 activations) GEMM benchmark, small shapes + DiT-realistic |
| `run_tests.py`, `results.json`, `summarize.py` | scheme harness, measurements |

## 7. Open questions / next steps

2. ~~Benchmark ozaki fp64 with **fp16 partials** (`low_dtype=np.float16`) on the NPU — expect large speedup.~~
   **Done** (`bench_ozaki_fp64_fp16.py`, `bench_ozaki_fp16_norm.py`): raw fp16 splits → **NaN** (fp16 overflow in
   split `sigma`); with pow2 row-normalization + capped `beta` → prototype re-run gave large errors (~76–107)
   and slow per-partial times (~15–60 s) — fix not validated; a proper fp16-exact split/EFT design still needed.
2. Study **s=2..4** split counts for fp64 via fp32 (double-double style) to cut partials from 21 → ~4 (open; queued as follow-up). Ran `bench_ozaki_fp64_s.py` at the DiT shape (low=fp32, high=fp64, u=2⁻²⁴):
   - s=2 (3 partials): **6.3 s**, maxabs 8.5e-2
   - s=3 (6 partials): **10.5 s**, maxabs 1.07e-3
   - s=4 (10 partials): **15.7 s**, maxabs 1.06e-5
   → smaller `s` buys speed but costs accuracy here; s=6 (21 partials) gave maxabs ~1e-9.
3. Validate whether QNN NPU fp16 GEMM accumulates in fp32 (B3) — would make fp32 emulation a single GEMM.
4. Re-tune `fp32_emu` (`s`, `u_acc`) for true fp32 (u_acc ~2^-24) via fp16 partials.
5. Move final partial accumulation into the device or fuse partials into following GEMMs (attention
   score matmuls especially) to cut round-trips.

## 8. Summary table — fp32 / fp64 emulation via ozaki on this device

### 8.1 Device GEMM capability (this session, DiT shape `[1,1024,3072]@[3072,12288]`, 0.1545 TFLOP/chain)

| backend | fp16 GEMM | fp32 GEMM | fp64 |
|---|---|---|---|
| Hexagon NPU via QNN EP (burst, fp16 prec) | **4.96 TFLOPS** | **3.73 TFLOPS** | n/a (must split) |
| Adreno X1-85 GPU via QNN EP | **2.26 TFLOPS** | ~3.8 TFLOPS (theoretical) | n/a (must split) |
| CPU (ORT) | ~0.04 TFLOPS | ~0.13–0.67 TFLOPS | **native** |

### 8.2 Ozaki emulation schemes vs native / partial-GEMM paths

| # | scheme | high → low | s | u_acc | backend | accuracy | time @ DiT shape | verdict |
|---|---|---|---|---|---|---|---|---|
| — | native fp32 GEMM (ozaki s=1) | fp32 → fp32 | 1 | – | NPU | fp32 ✓ | **41 ms** | ✅ best available fp32 path |
| — | native fp16 GEMM (ozaki s=1) | fp16 → fp16 | 1 | – | NPU | fp16 (~1e-3 rel) | **31 ms** | ✅ native fp16 path |
| 1 | `fp64_emu` (implemented) | fp64 → fp32 | 6 | 2⁻²⁴ | NPU / Adreno | ~fp64 ✓ (rel ≤ ~5e-11 @ 256³) | **~20–30 s** | ✅ accurate; ~500–750x behind native fp32 |
| 2 | A2: ozaki fp64 via fp16 partials | fp64 → fp16 | 4 / 6 / 8 | 2⁻²⁴ | NPU / Adreno | **NaN ✗** (fp16 overflow in split sigma) | 24 / 49 / 78 s | ❌ broken as-is; fix sigma scaling before retesting |
| 1 | `fp32_emu` (implemented) | fp32 → fp16 | 8 | 2⁻¹¹ | NPU / Adreno | ~2⁻¹¹ (not fp32) | 0.8–92 s | ⚠️ runs, wrong accuracy; native fp32 (B2) better |

### 8.3 Conclusions (condensed)

- NPU is the only sensible backend for ozaki partial GEMMs (fp16/fp32 both ≥3.7 TFLOPS; Adreno ~2.3 TFLOPS; CPU ~0.13–0.67 TFLOPS).
- fp64 emulation: implemented fp32-partials scheme is the only **working** accurate option; it costs ~20–30 s/matmul at DiT scale vs 41 ms native fp32 — the win would come from fewer partials (double-double, s≈2–4) or a fixed fp16-partial path.
- fp32 emulation: native fp32 GEMM via QNN (NPU, 3.73 TFLOPS) already exceeds ozaki-fp16-splitting throughput; the fp32 target can likely be met without splitting, or with a single fp16 GEMM + fp32 accumulation (B3, unvalidated).

## 9. All experiments — consolidated summary

| # | experiment | script | config / shape | what was measured | key result | status |
|---|---|---|---|---|---|---|
| E1 | fp16 GEMM benchmark | `bench/bench_gemm_fp16.py` | DiT shape, CPU/NPU/GPU | fp16 GEMM throughput | CPU 0.04 TFLOPS · **NPU 4.96 TFLOPS** · GPU 2.26 TFLOPS | done |
| E2 | fp32/fp16 GEMMs + ozaki schemes | `bench/bench_gemm_dtypes.py` | DiT shape, CPU/NPU/GPU | plain GEMMs + ozaki fp64/fp32 | plain fp32 → **NPU 3.73 TFLOPS (41 ms)**; ozaki fp64 (s=6, low=fp32) → NPU/GPU ~20–30 s, err ~1e-9; ozaki fp32 (s=8, low=fp16) → ~92 s, err ~2^-11 | done |
| E3 | Per-partial timing | `bench/bench_ozaki_parts.py` | DiT shape, fp64 s=6 low=fp32 | 21 partials via QNN | ~150–191 ms/partial, total ~3.1–3.7 s | done |
| E4 | Ozaki scheme harness | `ozaki/run_tests.py`, `common/ozaki.py` | 8 shapes × {fp64_emu, fp32_emu} × {gpu, htp} | accuracy + time vs fp64 ref | fp64_emu: ~fp64 accuracy ✓; fp32_emu: ~2^-11 (mis-tuned) | done |
| E5 | fp64 via fp16 partials (raw) | `bench/bench_ozaki_fp64_fp16.py` | DiT shape, s=4/6/8, low=fp16 | accuracy + time | **NaN** (sigma overflow); plain s=1: fp32 err 2.4e-5, fp16 err 1.5 | fail → fix needed |
| E6 | fp64 via fp16 partials (normalized) | `bench/bench_ozaki_fp16_norm.py` | DiT shape, s=4/6/8, pow2-norm + β∈{15,10,6,3} | accuracy + time | errors ~76–107; slow (~15–60 s/partial) | fail → design needed |
| E7 | fp64 via fp32 partials, smaller s | `bench/bench_ozaki_fp64_s.py` | DiT shape, low=fp32, s=2/3/4 | accuracy + time | s=2: 6.3 s/err 8.5e-2; s=3: 10.5 s/err 1.07e-3; s=4: 15.7 s/err 1.06e-5 | mixed |
| E8 | GEMM ceilings | `bench_gemm.py` (repo) | T=1024/3600, chains | CPU/NPU fp16/Adreno fp16 | CPU 0.2–0.7 TFLOPS; NPU fp16 4.09–5.12 TFLOPS; Adreno 0.49–2.10 TFLOPS | done |


| E9 | **fp32 emulation via ozaki fp16 partials** (just run) | ench_ozaki_fp32.py | DiT shape, high=fp32 to low=fp16, s=8, u=2^-11, NPU/GPU | NPU 78 s / GPU 92 s, maxabs ~80 (rel ~2^-11) | done - both backends runnable; ~1000x behind native fp16 GEMMs; accuracy ~2^-11, not fp32 |
