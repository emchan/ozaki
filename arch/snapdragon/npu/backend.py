"""Snapdragon backend: Hexagon NPU (via ONNX Runtime QNN EP).

Project layout:
    ozaki/
      common/            -> the ozaki method core
      arch/snapdragon/   -> architecture-specific backends
        cpu/              -> integrated CPU
        gpu/              -> Adreno GPU
        npu/              -> Hexagon NPU (this module)
      bench/             -> experiment / benchmark scripts
      run_tests.py       -> scheme harness
"""
def create_backend(backend: str, M: int, K: int, N: int, dtype=None, performance_mode: str = "burst", enable_fp16_precision: bool = True):
    """backend: 'npu' -> QNN EP with Hexagon backend path (fp16 precision enabled by default)."""
    if backend != "npu":
        raise ValueError(f"backend npu expected, got {backend}")
    from common.ozaki import QNNSession

    return QNNSession(M, K, N, dtype, "npu")
