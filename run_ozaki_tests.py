#!/usr/bin/env python3
"""High-level wrapper: create/use a venv-<arch> and run platform Ozaki tests."""
import os
import sys
import platform
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCH = platform.machine().lower().replace(" ", "-")
VENV = os.path.join(ROOT, f"venv-{ARCH}")

if sys.platform == "win32":
    VENV_PY = os.path.join(VENV, "Scripts", "python.exe")
else:
    VENV_PY = os.path.join(VENV, "bin", "python")

PLATFORM_TESTS = {
    "Windows": [
        "common/run_tests.py",
        "arch/amd/run_tests_amd.py",
        "arch/amd/run_tests_npu_amd.py",
    ],
    "Darwin": [
        "arch/apple/run_eval_mac.py",
        "arch/apple/run_dit_mac.py",
    ],
    "Linux": [],
}


def ensure_venv():
    if os.path.exists(VENV_PY):
        return VENV_PY
    print(f"Creating virtual environment: {VENV}")
    try:
        subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
    except Exception as e:
        print(f"Failed to create venv ({e}); falling back to {sys.executable}")
        return sys.executable
    if not os.path.exists(VENV_PY):
        print("venv python not found; falling back to current interpreter.")
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


def main():
    system = platform.system()
    tests = PLATFORM_TESTS.get(system, [])
    print(f"Platform: {system}  Arch: {ARCH}")
    if not tests:
        print("No platform-specific Ozaki tests to run on this platform.")
        return 0
    py = ensure_venv()
    ok = True
    for t in tests:
        ok = run(py, t) and ok
    print("\nAll tests finished." if ok else "\nSome tests failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
