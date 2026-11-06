#!/usr/bin/env bash
# Aggregate the three seeds: mean ± std of every metric -> runs/aggregate/summary.{json,md}
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
CONFIG=configs/dpo.yaml
RUN_NAME=$(python3 -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['run_name'])")
RUNS_DIR=$(python3 -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['runs_dir'])")
python3 -m saferl.aggregate --runs "$RUNS_DIR/${RUN_NAME}_seed42" "$RUNS_DIR/${RUN_NAME}_seed43" "$RUNS_DIR/${RUN_NAME}_seed44" --out "$RUNS_DIR/aggregate"
