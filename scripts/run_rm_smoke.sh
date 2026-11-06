#!/usr/bin/env bash
# Engineering smoke for all four score-model trainings (0.5B, tiny subset, 4 steps each): train + eval + summary.
#   bash scripts/run_rm_smoke.sh                 # all archs
#   bash scripts/run_rm_smoke.sh shared          # one arch
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export WORLD_SIZE="${WORLD_SIZE:-1}"
test -f artifacts/smoke/length_limits.json || python3 scripts/prepare_data.py --config configs/data_smoke.yaml
ARCHS="${1:-scalar shared separate_reward separate_cost}"
MODEL_OVR=""; TAG=qwen2.5-0.5b
if [ -n "${RM_SMOKE_MODEL:-}" ]; then MODEL_OVR="model=${RM_SMOKE_MODEL}"; TAG=1p5b; fi   # e.g. RM_SMOKE_MODEL=Qwen/Qwen2.5-1.5B-Instruct on the GPU box
for ARCH in $ARCHS; do
  RUN=rm_smoke_${ARCH}_${TAG}
  python3 -m saferl.train_rm --config configs/rm_smoke.yaml --seed 42 --set arch="$ARCH" run_name="$RUN" $MODEL_OVR
  python3 -m saferl.eval_rm  --config configs/rm_smoke.yaml --run_dir "runs/${RUN}_seed42" --set arch="$ARCH" run_name="$RUN" $MODEL_OVR
  echo "--- smoke eval summary ($ARCH) ---"; cat "runs/${RUN}_seed42/eval/eval_summary.md"
done
