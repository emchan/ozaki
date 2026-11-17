#!/usr/bin/env python3
"""High-level wrapper: run the platform-appropriate Ozaki tests and refresh results."""
import os
import sys
import platform
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

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

def run(script):
    path = os.path.join(ROOT, script)
    if not os.path.exists(path):
        print(f"[skip] missing: {script}")
        return True
    print(f"\n=== Running {script} ===")
    res = subprocess.run([PY, path], cwd=ROOT)
    if res.returncode != 0:
        print(f"[FAIL] {script} (exit {res.returncode})")
    else:
        print(f"[OK]   {script}")
    return res.returncode == 0

def main():
    system = platform.system()
    tests = PLATFORM_TESTS.get(system, [])
    print(f"Platform: {system}")
    if not tests:
        print("No platform-specific Ozaki tests to run on this platform.")
        return 0
    ok = True
    for t in tests:
        ok = run(t) and ok
    print("\nAll tests finished." if ok else "\nSome tests failed.")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
