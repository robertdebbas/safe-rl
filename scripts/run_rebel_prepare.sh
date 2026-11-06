#!/usr/bin/env bash
# Build the fixed REBEL prompt manifests from UNUSED official PKU-SafeRLHF train prompts (train 9,054 / calibration 256 / monitoring 256),
# prompt-disjoint from every existing split, all official test prompts and the old GRPO manifests. CPU only (~1-2 min).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
ART="${1:-artifacts}"
python3 -m saferl.rebel_data "$ART"
