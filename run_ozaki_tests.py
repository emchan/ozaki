#!/usr/bin/env python3
"""High-level wrapper: create/use a venv-<arch> and run platform Ozaki tests."""
import os
import sys
import platform
import subprocess
import json
import shutil
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCH = platform.machine().lower().replace(" ", "-")
VENV = os.path.join(ROOT, f"venv-{ARCH}")

if sys.platform == "win32":
    VENV_PY = os.path.join(VENV, "Scripts", "python.exe")
else:
    VENV_PY = os.path.join(VENV, "bin", "python")

def _is_arm_machine(machine):
    m = (machine or "").lower().replace(" ", "-")
    return m in ("arm64", "aarch64", "armv7l", "armv8l") or "arm" in m or "aarch" in m


def platform_tests(system=None, machine=None):
    """Return the Ozaki test scripts appropriate for the current platform/arch."""
    if system is None:
        system = platform.system()
    if machine is None:
        machine = platform.machine()
    if system == "Darwin":
        return [
            "arch/apple/run_eval_mac.py",
            "arch/apple/w8a16_test.py",
        ]
    if system == "Linux":
        return []
    if system == "Windows":
        # Snapdragon / other ARM Windows parts use the QNN (htp/gpu) harness;
        # AMD (Ryzen AI) parts use the Ryzen AI harnesses.
        if _is_arm_machine(machine):
            return [
                "common/run_tests.py",
                "arch/snapdragon/w8a16_test.py",
            ]
        return [
            "arch/amd/run_tests_amd.py",
            "arch/amd/run_tests_npu_amd.py",
        ]
    return []


PLATFORM_TESTS = {
    "Darwin": platform_tests("Darwin"),
    "Linux": [],
    "Windows": platform_tests("Windows", platform.machine()),
}


def _venv_usable(py):
    try:
        subprocess.run([py, "-c", "import numpy, onnx, onnxruntime"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def ensure_venv():
    if os.path.exists(VENV_PY) and _venv_usable(VENV_PY):
        return VENV_PY
    if os.path.exists(VENV):
        try:
            shutil.rmtree(VENV)
        except Exception as e:
            print(f"Failed to remove unusable venv ({e}); falling back to {sys.executable}")
            return sys.executable
    print(f"Creating virtual environment: {VENV}")
    try:
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", VENV], check=True)
    except Exception as e:
        print(f"Failed to create venv ({e}); falling back to {sys.executable}")
        return sys.executable
    if not os.path.exists(VENV_PY) or not _venv_usable(VENV_PY):
        print("venv python not found or unusable; falling back to current interpreter.")
        return sys.executable
    return VENV_PY


def run(py, script):
    path = os.path.join(ROOT, script)
    if not os.path.exists(path):
        print(f"[skip] missing: {script}")
        return True
    print(f"\n=== Running {script} with {os.path.basename(py)} ===")
    res = subprocess.run([py, path], cwd=ROOT)
    if res.returncode != 0:
        print(f"[FAIL] {script} (exit {res.returncode})")
    else:
        print(f"[OK]   {script}")
    return res.returncode == 0


RESULT_FILES = [
    "results/results.json",
    "arch/amd/results/results_amd.json",
    "arch/amd/results/results_npu_amd.json",
    "arch/apple/results/results_mac.json",
    "arch/apple/results/w8a16_results.json",
    "arch/snapdragon/results/w8a16_results.json",
]


PRECISION_MAP = {
    "fp64_emu": ("fp64", "fp32"),
    "fp32_emu": ("fp32", "fp16"),
    "w8a16": ("fp16", "w8a16"),
}

_DEVICES = []
_OVERALL_ROWS = []


def _device_for_result_file(rel):
    low = rel.lower()
    if "apple" in low:
        return "apple"
    if "amd" in low:
        return "amd"
    if "snapdragon" in low:
        return "snapdragon"
    if rel == "results/results.json":
        return "host"
    return os.path.splitext(os.path.basename(rel))[0]


def _product_for_device(device):
    return {
        "apple": "Apple M4",
        "snapdragon": "Snapdragon X1P64100",
        "amd": "AMD Ryzen 7 7840HS",
        "host": "host",
    }.get(device, device)


def collect_results():
    global _DEVICES, _OVERALL_ROWS
    data = {}
    devices = []
    overall_rows = []
    for rel in RESULT_FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        try:
            rows = json.load(open(path))
        except Exception:
            continue
        device = _device_for_result_file(rel)
        for r in rows:
            scheme = r.get("scheme")
            backend = r.get("backend")
            if backend == "htp":
                backend = "npu"
            if scheme is None or backend is None:
                continue
            t = r.get("ozaki_time_ms", r.get("ms"))
            m = r.get("metrics_vs_fp64", {})
            err = m.get("max_abs", r.get("maxabs"))
            if t is None or err is None:
                continue
            row_device = r.get("device", device)
            row_product = r.get("product", _product_for_device(row_device))
            data.setdefault((row_device, scheme, backend), []).append((float(t), float(err)))
            devices.append(row_device)
            overall_rows.append(dict(r, device=row_device, product=row_product, source=rel))

    _DEVICES = sorted(set(d for d in devices if d != "host"))
    _OVERALL_ROWS = overall_rows

    # Save the overall (combined) results to results/.
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    overall_path = os.path.join(out_dir, "overall_results.json")
    with open(overall_path, "w") as f:
        json.dump(overall_rows, f, indent=2)
    # Convenience alias.
    with open(os.path.join(out_dir, "overall.json"), "w") as f:
        json.dump(overall_rows, f, indent=2)

    schemes = sorted({k[1] for k in data})
    backends = sorted({k[2] for k in data})
    return schemes, backends, data


def print_summary():
    schemes, backends, data = collect_results()
    if not schemes or not backends:
        print("\nNo result data available for summary.")
        return
    devices = _DEVICES or ["unknown"]
    # Build aggregated values
    agg = {}
    for (device, scheme, backend), vals in data.items():
        ts = [v[0] for v in vals]
        es = [v[1] for v in vals]
        agg[(device, scheme, backend)] = (sum(ts) / len(ts), sum(es) / len(es))

    # Column widths
    colw = 22
    device_w = 14
    product_w = 22
    native_w = 12
    target_w = 10
    speedup_w = 24
    header = ("device".ljust(device_w) +
              "native".ljust(native_w) + "target (emu)".ljust(target_w + 6) +
              "".join(b.center(colw) for b in backends) +
              "best speedup vs CPU".center(speedup_w) +
              "product".ljust(product_w))
    print("\n" + "=" * len(header))
    print("Speed (ms avg) / Error (max abs avg)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for device in devices:
        for scheme in schemes:
            target, native = PRECISION_MAP.get(scheme, (scheme, "?"))
            product = _product_for_device(device)
            row = (device.ljust(device_w) +
                   native.ljust(native_w) + target.ljust(target_w + 6))
            for backend in backends:
                v = agg.get((device, scheme, backend))
                if v is None:
                    cell = "n/a"
                else:
                    cell = f"{v[0]:.1f} ms / {v[1]:.2e}"
                row += cell.center(colw)
            # Best-case speedup vs CPU emulation on GPU/NPU.
            cpu_v = agg.get((device, scheme, "cpu"))
            best = None
            best_dev = None
            if cpu_v is not None:
                for b in ("gpu", "npu"):
                    v = agg.get((device, scheme, b))
                    if v is not None and v[0] > 0:
                        speedup = cpu_v[0] / v[0]
                        if best is None or speedup > best:
                            best = speedup
                            best_dev = b
            if best is None:
                speedup_cell = "n/a"
            else:
                speedup_cell = f"{best:.2f}x ({best_dev})"
            row += speedup_cell.center(speedup_w)
            row += product.ljust(product_w)
            print(row)
    print("=" * len(header))

    print_cpu_native_capability()


def print_cpu_native_capability():
    """Benchmark native CPU fp16/fp32/fp64 GEMM capability as a reference."""
    try:
        import numpy as np
    except Exception as e:
        print(f"\nCPU native capability: numpy unavailable ({e})")
        return
    m, k, n = 256, 256, 256
    rng = np.random.default_rng(0)
    print("\n" + "=" * 60)
    print("CPU native GEMM capability (reference)")
    print("=" * 60)
    print(f"{'precision':<12}{'time (ms)':<14}{'TFLOPS':<10}")
    print("-" * 60)
    for name, dtype in (("fp16", np.float16), ("fp32", np.float32), ("fp64", np.float64)):
        try:
            A = (rng.standard_normal((m, k)) * 0.5).astype(dtype)
            B = (rng.standard_normal((k, n)) * 0.5).astype(dtype)
            # Warmup.
            _ = A @ B
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                _ = A @ B
                times.append(time.perf_counter() - t0)
            t = sorted(times)[len(times) // 2]
            tflops = (2.0 * m * k * n / t) / 1e12
            print(f"{name:<12}{t * 1000:<14.2f}{tflops:<10.3f}")
        except Exception as e:
            print(f"{name:<12}{'error':<14}{str(e)}")
    print("=" * 60)


def main():
    system = platform.system()
    machine = platform.machine()
    tests = platform_tests(system, machine)
    print(f"Platform: {system}  Arch: {ARCH}  Machine: {machine}")
    if not tests:
        print("No platform-specific Ozaki tests to run on this platform.")
        print_summary()
        return 0
    py = ensure_venv()
    ok = True
    for t in tests:
        ok = run(py, t) and ok
    print("\nAll tests finished." if ok else "\nSome tests failed.")
    print_summary()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
