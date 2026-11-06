#!/usr/bin/env bash
# Matched constrained-GRPO comparison (same prompts / calibration / d / dual as REBEL). PREPARED ONLY -- launch requires approval.
#   bash scripts/run_grpo_matched.sh shared 42 [--set stop_after_batches=100]
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?arm: shared | separate}"; SEED="${2:-42}"; shift 2 || true
export PYTHONPATH=src
export WORLD_SIZE="${WORLD_SIZE:-1}"
case "$ARM" in shared) CONFIG=configs/grpo_matched_shared_constrained.yaml;; separate) CONFIG=configs/grpo_matched_separate_constrained.yaml;; *) echo "unknown arm"; exit 1;; esac
python3 -m saferl.grpo_matched --config "$CONFIG" --seed "$SEED" "$@"
