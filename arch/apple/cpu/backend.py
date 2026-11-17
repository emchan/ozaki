"""macOS Apple Silicon CPU backend (PyTorch CPU)."""
import time
import numpy as np
import torch


class Backend:
    def __init__(self, backend, M, K, N, dtype, model_dir="/tmp"):
        if backend != "cpu":
            raise ValueError(f"cpu backend expected, got {backend}")
        self.backend = backend
        self.M, self.K, self.N = M, K, N
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float16, np.float32):
            raise ValueError(f"Unsupported dtype for Apple backends: {self.dtype}")
        self.torch_dtype = torch.float16 if self.dtype == np.float16 else torch.float32
        self.device = "cpu"

    def run(self, A_, B_):
        Aa = np.asarray(A_, dtype=self.dtype)
        Bb = np.asarray(B_, dtype=self.dtype)
        ta = torch.from_numpy(Aa).to(self.torch_dtype)
        tb = torch.from_numpy(Bb).to(self.torch_dtype)
        with torch.no_grad():
            tc = ta @ tb
        out = tc.detach().cpu().numpy()
        return np.array(out, dtype=self.dtype)


def create_backend(backend, M, K, N, dtype=np.float32, model_dir="/tmp"):
    return Backend(backend, M, K, N, dtype, model_dir=model_dir)
