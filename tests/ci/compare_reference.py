"""Regression test: compare an LPC3D run with stored reference results.

Every reference file in <reference_directory> must exist in <run_directory>
with the same shape, and its values must agree within a tolerance:

    |run - ref| <= ATOL_FACTOR * max|ref| + RTOL * |ref|

LPC3D fixes its random seed, so results are bit-for-bit identical from run
to run and independent of the number of threads. Between different
installations (Python/NumPy versions, CPU types) only rounding-level
differences are expected (about 1e-10 relative in our tests), so the
tolerance below is tight and any change in the physics will be caught.

Usage: python3 compare_reference.py <run_directory> <reference_directory>
"""
import sys
from pathlib import Path

import numpy as np

RTOL = 1e-5
ATOL_FACTOR = 1e-6


def main():
    run_dir = Path(sys.argv[1])
    ref_dir = Path(sys.argv[2])
    ref_files = sorted(p for p in ref_dir.iterdir() if p.is_file())
    if not ref_files:
        print(f"No reference files found in {ref_dir}")
        sys.exit(1)

    errors = []
    for ref_path in ref_files:
        run_path = run_dir / ref_path.name
        if not run_path.is_file():
            errors.append(f"{ref_path.name}: missing from the run")
            continue
        ref = np.loadtxt(ref_path, ndmin=1)
        new = np.loadtxt(run_path, ndmin=1)
        if new.shape != ref.shape:
            errors.append(f"{ref_path.name}: shape {new.shape}, expected {ref.shape}")
            continue
        scale = np.max(np.abs(ref)) if ref.size else 0.0
        atol = ATOL_FACTOR * scale
        diff = np.abs(new - ref)
        bad = ~(diff <= atol + RTOL * np.abs(ref))
        if np.any(bad):
            i = np.unravel_index(np.argmax(np.where(bad, diff, -1)), ref.shape)
            errors.append(
                f"{ref_path.name}: {int(bad.sum())} value(s) outside tolerance, "
                f"largest at index {tuple(int(k) for k in i)}: "
                f"got {new[i]:.10g}, reference {ref[i]:.10g}"
            )

    if errors:
        print(f"Regression test FAILED against {ref_dir}:")
        for e in errors:
            print("  -", e)
        print(
            "If this change in the results is intended, regenerate the "
            "references with tests/ci/update_reference.sh and commit them."
        )
        sys.exit(1)
    print(f"Regression test passed: {len(ref_files)} file(s) match {ref_dir}")


if __name__ == "__main__":
    main()
