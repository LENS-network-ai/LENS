#!/usr/bin/env bash
# Fetches the official WiKG repo (https://github.com/WonderLandxD/WiKG,
# Li et al., CVPR 2024) for use as a learned-graph baseline comparison.
#
# The upstream repo ships no LICENSE file, so its code is NOT vendored here:
# baselines/WiKG_original/ is git-ignored and this script re-materializes it
# locally. baselines/wikg.py imports only upstream's model.py (unmodified);
# compatibility with newer torch_geometric versions is handled at import
# time in the wrapper, so no patch file is needed.
#
# Usage:
#   bash baselines/setup_wikg_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/WiKG_original"
# Pinned to the commit baselines/wikg.py was written against.
PIN_COMMIT="eb3144fb8267f525c57e71ced5c5b2b6a1c42a93"

if [ -d "${TARGET_DIR}" ]; then
    echo "Removing existing ${TARGET_DIR} before re-cloning..."
    rm -rf "${TARGET_DIR}"
fi

echo "Cloning WiKG @ ${PIN_COMMIT}..."
git clone https://github.com/WonderLandxD/WiKG.git "${TARGET_DIR}"
git -C "${TARGET_DIR}" checkout "${PIN_COMMIT}"

echo "Done. WiKG baseline ready at ${TARGET_DIR}"
