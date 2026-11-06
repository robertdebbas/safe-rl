#!/usr/bin/env bash
# 1) generate the 256 x 4 pi_0 calibration trajectories once (shared by both constrained arms);
# 2) score them with the arm's frozen cost model and write d_<arm>.    bash scripts/run_grpo_calibrate.sh shared ; ... separate
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?arm: shared | separate}"
export PYTHONPATH=src
CONFIG="configs/grpo_${ARM}_constrained.yaml"
test -f artifacts/grpo/train_prompts.jsonl || bash scripts/run_grpo_prepare.sh
python3 -m saferl.grpo_calibrate --config "$CONFIG" --stage both
