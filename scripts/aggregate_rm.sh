#!/usr/bin/env bash
# Aggregate seeds 42/43/44 of one architecture: bash scripts/aggregate_rm.sh scalar
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH="${1:?arch}"
export PYTHONPATH=src
RUN_NAME=$(python3 -c "from saferl.utils import load_config;print(load_config('configs/rm_${ARCH}.yaml')['run_name'])")
python3 -m saferl.aggregate_rm --runs "runs/${RUN_NAME}_seed42" "runs/${RUN_NAME}_seed43" "runs/${RUN_NAME}_seed44" --out "runs/aggregate_rm_${ARCH}"
