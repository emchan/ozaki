"""Backend adapters used by the Ozaki scheme harness on Cloud TPUs.

The high-precision splitting/recombination stays in :mod:`common.ozaki`; these
sessions only provide the low-precision ``A @ B`` partial products.  ``tpu``
runs them on a JAX TPU device, ``cpu`` is included as the reference backend so
the summary can compute a speedup against CPU emulation.
"""
import os

import numpy as np


def tpu_devices():
    """Return the JAX devices whose platform is a TPU (empty if unavailable)."""
    try:
        import jax
    except Exception:
        return []
    try:
        return [d for d in jax.devices() if str(d.platform).lower() == "tpu"]
    except Exception:
        return []


def is_tpu_available():
    """True when JAX can actually see at least one TPU device."""
    return bool(tpu_devices())


def tpu_product():
    """Best-effort product name for the current Cloud TPU VM."""
    accelerator = os.environ.get("TPU_ACCELERATOR_TYPE", "").strip()
    if accelerator:
        family = accelerator.split("-")[0]
        return f"Google Cloud TPU {family}"
    devices = tpu_devices()
    if devices:
        kind = getattr(devices[0], "device_kind", "") or ""
        if kind:
            return f"Google Cloud TPU {kind}"
    return "Google Cloud TPU"


class TPUSession:
    """Run low-precision matmul partials on a JAX TPU device."""

    def __init__(self, M, K, N, dtype=np.float32, device=None):
        import jax
        import jax.numpy as jnp

        self.M, self.K, self.N = M, K, N
        self.npdt = np.dtype(dtype)
        devices = tpu_devices()
        if not devices:
            raise RuntimeError("JAX cannot see a TPU device")
        self.device = device if device is not None else devices[0]
        # HIGHEST keeps fp32 partials in fp32 on TPUs (the default may use
        # bf16), which matters for the fp64_emu scheme's accuracy.
        self._dot = jax.jit(
            lambda a, b: jnp.dot(a, b, precision=jax.lax.Precision.HIGHEST)
        )

    def run(self, A, B):
        import jax

        a = jax.device_put(np.asarray(A, dtype=self.npdt), self.device)
        b = jax.device_put(np.asarray(B, dtype=self.npdt), self.device)
        c = self._dot(a, b)
        c.block_until_ready()  # include device execution in the measured time
        return np.asarray(c, dtype=self.npdt)


class CPUSession:
    """Reference backend: low-precision GEMM on the host CPU."""

    def __init__(self, M, K, N, dtype=np.float32):
        self.M, self.K, self.N = M, K, N
        self.npdt = np.dtype(dtype)

    def run(self, A, B):
        a = np.asarray(A, dtype=self.npdt)
        b = np.asarray(B, dtype=self.npdt)
        return np.asarray(a @ b, dtype=self.npdt)


def create_backend(backend, M, K, N, dtype=np.float32, model_dir=None):
    """Create a session compatible with ``common.ozaki.ozaki_matmul``."""
    if backend == "tpu":
        return TPUSession(M, K, N, dtype)
    if backend == "cpu":
        return CPUSession(M, K, N, dtype)
    raise ValueError(f"unknown backend {backend}")
