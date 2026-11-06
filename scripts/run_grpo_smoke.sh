#!/usr/bin/env bash
# Engineering smoke of the GRPO stage with 0.5B models: manifests -> calibration (shared, separate) -> 3 rollout batches per arm,
# then an interrupted + resumed scalar run to exercise resume.   bash scripts/run_grpo_smoke.sh [scalar shared separate]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
ART=artifacts/smoke
test -f $ART/length_limits.json || python3 scripts/prepare_data.py --config configs/data_smoke.yaml
python3 -c "from saferl.grpo_data import build_manifests; build_manifests('$ART', n_calibration=4, n_monitoring=4)"
ARMS="${1:-scalar shared separate}"
SMOKE_RM=runs/rm_smoke
for ARM in $ARMS; do
  case "$ARM" in
    scalar)   SC="scorer_checkpoints={scalar: ${SMOKE_RM}_scalar_qwen2.5-0.5b_seed42/checkpoints/best}"; CAL="";;
    shared)   SC="scorer_checkpoints={shared: ${SMOKE_RM}_shared_qwen2.5-0.5b_seed42/checkpoints/best}"; CAL="calibration_file=$ART/grpo/calibration_shared.json";;
    separate) SC="scorer_checkpoints={reward: ${SMOKE_RM}_separate_reward_qwen2.5-0.5b_seed42/checkpoints/best, cost: ${SMOKE_RM}_separate_cost_qwen2.5-0.5b_seed42/checkpoints/best}"; CAL="calibration_file=$ART/grpo/calibration_separate.json";;
  esac
  if [ -n "$CAL" ]; then
    python3 -m saferl.grpo_calibrate --config configs/grpo_smoke.yaml --stage both --set arm="$ARM" "$SC"
  fi
  rm -rf "runs/grpo_smoke_${ARM}_seed42"
  python3 -m saferl.train_grpo --config configs/grpo_smoke.yaml --seed 42 --set arm="$ARM" run_name="grpo_smoke_${ARM}" "$SC" $CAL
  echo "--- smoke summary ($ARM) ---"; python3 -c "import json;print(json.dumps(json.load(open('runs/grpo_smoke_${ARM}_seed42/train_summary.json')), indent=1))" | head -40
done
