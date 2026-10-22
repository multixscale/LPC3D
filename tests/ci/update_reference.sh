#!/usr/bin/env bash
# Regenerate the reference results used by the CI regression tests.
# Run this only when a change in the results is intended (e.g. a bug fix or a
# change of model), then review and commit tests/ci/reference/.
#
# Usage: tests/ci/update_reference.sh [python_executable]
set -euo pipefail

PY="${1:-python3}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for case in bulk supercapacitor; do
    run="$TMP/$case"
    mkdir -p "$run"
    cp "$REPO/tests/ci/$case/lattice_gas.inpt" "$run/"
    if [ "$case" = "supercapacitor" ]; then
        cp "$REPO/examples/short-test/psd.txt" \
           "$REPO/examples/short-test/pore_dens_freq_2neg.txt" "$run/"
    fi
    (cd "$run" && "$PY" "$REPO/LPC3D.py" -i lattice_gas.inpt > log.lpc3d)

    ref="$REPO/tests/ci/reference/$case"
    rm -rf "$ref"
    mkdir -p "$ref"
    # All numerical outputs except the inputs, logs and memory report
    for f in "$run"/*.txt "$run"/*.dat; do
        name="$(basename "$f")"
        case "$name" in
            psd.txt|pore_dens_freq_2neg.txt|memory_usage.txt) continue ;;
        esac
        cp "$f" "$ref/"
    done
    echo "Updated $ref ($(ls "$ref" | wc -l) files)"
done
