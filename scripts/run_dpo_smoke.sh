#!/usr/bin/env bash
# Engineering smoke test: tiny data subset, Qwen2.5-0.5B-Instruct, 4 optimizer steps, train + eval + summary.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export WORLD_SIZE="${WORLD_SIZE:-1}"
CONFIG=configs/dpo_smoke.yaml
test -f artifacts/smoke/length_limits.json || python3 scripts/prepare_data.py --config configs/data_smoke.yaml
python3 -m saferl.train_dpo --config "$CONFIG" --seed 42 "$@"
python3 -m saferl.eval_dpo  --config "$CONFIG" --run_dir runs/smoke_qwen2.5-0.5b_seed42 "$@"
echo "--- smoke eval summary ---"; cat runs/smoke_qwen2.5-0.5b_seed42/eval/eval_summary.md
