"""Check that an LPC3D run produced the expected, valid output files.

Usage: python3 check_outputs.py <run_directory> <number_of_blocks>
"""
import sys
from pathlib import Path

import numpy as np


def main():
    run_dir = Path(sys.argv[1])
    nblocks = int(sys.argv[2])
    errors = []

    for b in range(1, nblocks + 1):
        expected = [
            f"density_{b}.txt",
            f"vacf_{b}.txt",
            f"Diffusion_coeff_x_{b}.txt",
            f"Diffusion_coeff_y_{b}.txt",
            f"Diffusion_coeff_z_{b}.txt",
            f"FT_signal-1-bloc-{b}.dat",
            f"Normalised_spectrum-1-bloc-{b}.dat",
        ]
        for name in expected:
            path = run_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty: {name}")
                continue
            try:
                data = np.loadtxt(path, ndmin=1)
            except ValueError as exc:
                errors.append(f"unreadable: {name} ({exc})")
                continue
            if not np.all(np.isfinite(data)):
                errors.append(f"NaN or Inf values in: {name}")

    if errors:
        print("Output check FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"Output check passed for {nblocks} block(s) in {run_dir}")


if __name__ == "__main__":
    main()
