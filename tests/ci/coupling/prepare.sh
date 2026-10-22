#!/usr/bin/env bash
# Prepare a reduced LAMMPS-LPC3D run for CI.
# Usage: tests/ci/coupling/prepare.sh <run_directory>
#
# The code and molecular files are taken from LAMMPS-LPC3D/ so the test always
# follows the real workflow. Only the size of the problem is reduced:
#   - 2 pore sizes (psd.txt), so 2 LAMMPS simulations
#   - 5000 water molecules instead of 50000 (lattice_gas.inpt, sites2cm.inpt)
#   - 100 NPT steps and 200 NVT steps instead of 10000 and 100000
#   - 10x10x30 lattice, no equilibration
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC="$REPO/LAMMPS-LPC3D"
CI="$REPO/tests/ci/coupling"
RUN="$1"

mkdir -p "$RUN"

for f in LAMMPS-LPC3D.py mpi4py_lammps.py fftool il.ff H2O.zmat CG1.xyz CG2.xyz \
         npt_in.lmp nvt_in.lmp; do
    cp "$SRC/$f" "$RUN/"
done
for f in lattice_gas.inpt psd.txt pore_freq_neg.txt pore_freq_pos.txt sites2cm.inpt; do
    cp "$CI/$f" "$RUN/"
done

# The workflow calls ./packmol in each simulation folder
cp "$(command -v packmol)" "$RUN/packmol"
chmod +x "$RUN/fftool" "$RUN/packmol"

# Shorten the MD runs, and fail loudly if the scripts change
shorten() {
    local file="$1" old="$2" new="$3"
    grep -q "^run $old\$" "$RUN/$file" || { echo "Expected 'run $old' in $file" >&2; exit 1; }
    sed -i "s/^run $old\$/run $new/" "$RUN/$file"
}
shorten npt_in.lmp 10000 100
shorten nvt_in.lmp 100000 200

echo "Reduced coupled test prepared in $RUN"
