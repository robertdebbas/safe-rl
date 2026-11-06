#!/usr/bin/env bash
# Train + evaluate one reward/cost-model architecture for one seed on one GPU.
#   bash scripts/run_rm.sh scalar 42
#   bash scripts/run_rm.sh shared 43
#   bash scripts/run_rm.sh separate_reward 44 ; bash scripts/run_rm.sh separate_cost 44
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH="${1:?arch: scalar | shared | separate_reward | separate_cost}"; SEED="${2:-42}"; shift 2 || true
export PYTHONPATH=src
export WORLD_SIZE="${WORLD_SIZE:-1}"
CONFIG="configs/rm_${ARCH}.yaml"
test -f "$CONFIG" || { echo "unknown arch $ARCH"; exit 1; }
test -f artifacts/length_limits.json || { echo "artifacts missing: run  python scripts/prepare_data.py --config configs/data.yaml  first"; exit 1; }
RUN_NAME=$(PYTHONPATH=src python3 -c "from saferl.utils import load_config;print(load_config('$CONFIG')['run_name'])")
python3 -m saferl.train_rm --config "$CONFIG" --seed "$SEED" "$@"
python3 -m saferl.eval_rm  --config "$CONFIG" --run_dir "runs/${RUN_NAME}_seed${SEED}" "$@"
