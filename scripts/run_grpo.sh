#!/usr/bin/env bash
# One policy-training seed of one arm (one prompt epoch), resumable: re-running the same command resumes from runs/<run>/resume/latest.pt
#   bash scripts/run_grpo.sh scalar 42 | shared 43 | separate 44
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?arm: scalar | shared | separate}"; SEED="${2:-42}"; shift 2 || true
export PYTHONPATH=src
export WORLD_SIZE="${WORLD_SIZE:-1}"
case "$ARM" in scalar) CONFIG=configs/grpo_scalar.yaml;; shared) CONFIG=configs/grpo_shared_constrained.yaml;; separate) CONFIG=configs/grpo_separate_constrained.yaml;; *) echo "unknown arm"; exit 1;; esac
test -f artifacts/grpo/train_prompts.jsonl || bash scripts/run_grpo_prepare.sh
python3 -m saferl.train_grpo --config "$CONFIG" --seed "$SEED" "$@"
