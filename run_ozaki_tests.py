#!/usr/bin/env python3
"""High-level wrapper: create/use a venv-<arch> and run platform Ozaki tests."""
import os
import sys
import platform
import subprocess
import json
import shutil

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
            "arch/apple/run_dit_mac.py",
        ]
    if system == "Linux":
        return []
    if system == "Windows":
        # Snapdragon / other ARM Windows parts use the QNN (htp/gpu) harness;
        # AMD (Ryzen AI) parts use the Ryzen AI harnesses.
        if _is_arm_machine(machine):
            return ["common/run_tests.py"]
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
    "arch/apple/results/results_dit_mac.json",
]


PRECISION_MAP = {
    "fp64_emu": ("fp64", "fp32"),
    "fp32_emu": ("fp32", "fp16"),
}

def collect_results():
    data = {}
    for rel in RESULT_FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        try:
            rows = json.load(open(path))
        except Exception:
            continue
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
            data.setdefault((scheme, backend), []).append((float(t), float(err)))

    schemes = sorted({k[0] for k in data})
    backends = sorted({k[1] for k in data})
    return schemes, backends, data


def print_summary():
    schemes, backends, data = collect_results()
    if not schemes or not backends:
        print("\nNo result data available for summary.")
        return
    # Build aggregated values
    agg = {}
    for (scheme, backend), vals in data.items():
        ts = [v[0] for v in vals]
        es = [v[1] for v in vals]
        agg[(scheme, backend)] = (sum(ts) / len(ts), sum(es) / len(es))

    # Column widths
    colw = 22
    native_w = 12
    target_w = 10
    header = "native".ljust(native_w) + "target (emu)".ljust(target_w + 6) + "".join(b.center(colw) for b in backends)
    print("\n" + "=" * len(header))
    print("Speed (ms avg) / Error (max abs avg)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for scheme in schemes:
        target, native = PRECISION_MAP.get(scheme, (scheme, "?"))
        row = native.ljust(native_w) + target.ljust(target_w + 6)
        for backend in backends:
            v = agg.get((scheme, backend))
            if v is None:
                cell = "n/a"
            else:
                cell = f"{v[0]:.1f} ms / {v[1]:.2e}"
            row += cell.center(colw)
        print(row)
    print("=" * len(header))



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
