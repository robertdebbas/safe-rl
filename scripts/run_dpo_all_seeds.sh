#!/usr/bin/env bash
# Run the three scientific seeds SEQUENTIALLY on one GPU, then aggregate.  (Do not launch casually: ~3 full runs.)
set -euo pipefail
cd "$(dirname "$0")/.."
for SEED in 42 43 44; do bash scripts/run_dpo.sh "$SEED" "$@"; done
bash scripts/aggregate.sh
