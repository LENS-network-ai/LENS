#!/usr/bin/env bash
# Fetches the original GraphLSurv repo (https://github.com/liupei101/GraphLSurv)
# for use as a survival-prediction baseline comparison, and applies the two
# small PyTorch/PyTorch-Geometric version-compatibility patches needed to run
# it against this repo's environment (see graphlsurv_pytorch_compat.patch for
# exactly what changed and why -- both are compatibility shims only, no
# behavior change to the original method).
#
# Usage:
#   bash baselines/setup_graphlsurv_baseline.sh
#
# This is NOT run automatically -- baselines/GraphLSurv_original/ is
# git-ignored rather than vendored, so this script is how you (re)materialize
# it locally before running training/train_graphlsurv_survival.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/GraphLSurv_original"
PATCH_FILE="${SCRIPT_DIR}/graphlsurv_pytorch_compat.patch"
# Pinned to the commit this patch was written against.
PIN_COMMIT="ab0d8b52c8217f4f927916816fa27913fd9b33d8"

if [ -d "${TARGET_DIR}" ]; then
    echo "Removing existing ${TARGET_DIR} before re-cloning..."
    rm -rf "${TARGET_DIR}"
fi

echo "Cloning GraphLSurv @ ${PIN_COMMIT}..."
git clone https://github.com/liupei101/GraphLSurv.git "${TARGET_DIR}"
git -C "${TARGET_DIR}" checkout "${PIN_COMMIT}"

echo "Applying compatibility patch..."
git -C "${TARGET_DIR}" apply "${PATCH_FILE}"

echo "Done. GraphLSurv baseline ready at ${TARGET_DIR}"
