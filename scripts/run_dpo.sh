#!/usr/bin/env bash
# Train one scientific seed, then evaluate its best-by-validation checkpoint on val/test splits.
#   bash scripts/run_dpo.sh 42            # seed 42
#   bash scripts/run_dpo.sh 43 --set per_device_batch_size=8
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:-42}"; shift || true
export PYTHONPATH=src
export WORLD_SIZE="${WORLD_SIZE:-1}"
CONFIG=configs/dpo.yaml
RUN_NAME=$(python3 -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['run_name'])")
RUNS_DIR=$(python3 -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['runs_dir'])")
test -f artifacts/length_limits.json || { echo "artifacts missing: run  python scripts/prepare_data.py --config configs/data.yaml  first"; exit 1; }
python3 -m saferl.train_dpo --config "$CONFIG" --seed "$SEED" "$@"
python3 -m saferl.eval_dpo  --config "$CONFIG" --run_dir "$RUNS_DIR/${RUN_NAME}_seed${SEED}" "$@"
