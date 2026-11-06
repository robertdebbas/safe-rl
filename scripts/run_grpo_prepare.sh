#!/usr/bin/env bash
# Build the fixed GRPO prompt manifests (train 9,054 / calibration 256 / monitoring 256) from the existing splits.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
ART="${1:-artifacts}"
python3 -c "from saferl.grpo_data import build_manifests; import json; print(json.dumps(build_manifests('$ART'), indent=1))"
