"""Check the LAMMPS side of a reduced LAMMPS-LPC3D run.

For every pore size in psd.txt, checks that the LAMMPS simulations ran
(NPT and NVT logs completed, centre-of-mass file written), and that the
densities passed back to LPC3D (all_densities.dat) are finite, non-negative
and not all zero.

Usage: python3 check_coupling.py <run_directory>
"""
import sys
from pathlib import Path

import numpy as np


def main():
    run_dir = Path(sys.argv[1])
    errors = []

    pore_sizes = np.unique(np.loadtxt(run_dir / "psd.txt", usecols=0, ndmin=1))

    for p in pore_sizes:
        sim = run_dir / f"simulation_{p:g}"
        if not sim.is_dir():
            # the folder name follows Python's str() of the pore size
            sim = run_dir / f"simulation_{float(p)}"
        if not sim.is_dir():
            errors.append(f"missing simulation folder for pore size {p:g}")
            continue
        for log in (sim / "log.lammps", sim / "NVT" / "log.lammps"):
            if not log.is_file() or "Total wall time" not in log.read_text():
                errors.append(f"LAMMPS run did not complete: {log.relative_to(run_dir)}")
        cm = sim / "NVT" / "CM_positions.xyz"
        if not cm.is_file() or cm.stat().st_size == 0:
            errors.append(f"missing or empty: {cm.relative_to(run_dir)}")

    dens_file = run_dir / "all_densities.dat"
    if not dens_file.is_file():
        errors.append("missing: all_densities.dat")
    else:
        dens = np.loadtxt(dens_file, ndmin=2)
        if dens.shape[0] != len(pore_sizes):
            errors.append(
                f"all_densities.dat has {dens.shape[0]} rows, expected {len(pore_sizes)}"
            )
        values = dens[:, 1:]
        if not np.all(np.isfinite(values)):
            errors.append("NaN or Inf in all_densities.dat")
        elif np.any(values < 0):
            errors.append("negative density in all_densities.dat")
        elif not np.any(values > 0):
            errors.append("all densities are zero in all_densities.dat")

    log = run_dir / "log.lpc3d"
    if log.is_file() and "still not equilibrated" in log.read_text():
        errors.append("LPC3D reported that the system is not equilibrated")

    if errors:
        print("Coupling check FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"Coupling check passed for {len(pore_sizes)} pore size(s) in {run_dir}")


if __name__ == "__main__":
    main()
