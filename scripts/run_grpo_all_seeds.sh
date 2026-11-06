#!/usr/bin/env bash
# Seeds 42/43/44 of one arm, sequentially on one GPU.  bash scripts/run_grpo_all_seeds.sh scalar
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?arm}"
for SEED in 42 43 44; do bash scripts/run_grpo.sh "$ARM" "$SEED"; done
