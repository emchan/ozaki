"""Snapdragon backend: Adreno integrated GPU (via ONNX Runtime QNN EP).

Project layout:
    ozaki/
      common/            -> the ozaki method core
      arch/snapdragon/   -> architecture-specific backends
        cpu/              -> integrated CPU
        gpu/              -> Adreno GPU (this module)
        npu/              -> Hexagon NPU
      bench/             -> experiment / benchmark scripts
      run_tests.py       -> scheme harness
"""
def create_backend(backend: str, M: int, K: int, N: int, dtype=None, performance_mode: str = "burst", enable_fp16_precision: bool = True):
    """backend: 'gpu' -> QNN EP with Adreno backend path (fp16 precision enabled by default)."""
    if backend != "gpu":
        raise ValueError(f"backend gpu expected, got {backend}")
    from common.ozaki import QNNSession

    return QNNSession(M, K, N, dtype, "gpu")
