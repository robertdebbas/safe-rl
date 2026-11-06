#!/usr/bin/env bash
# One policy-training seed of one REBEL arm (one prompt epoch), resumable: re-running the same command resumes from runs/<run>/resume/latest.pt
#   bash scripts/run_rebel.sh scalar 42 [--set stop_after_batches=100]
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?arm: scalar | shared | separate}"; SEED="${2:-42}"; shift 2 || true
export PYTHONPATH=src
export WORLD_SIZE="${WORLD_SIZE:-1}"
case "$ARM" in scalar) CONFIG=configs/rebel_scalar.yaml;; shared) CONFIG=configs/rebel_shared_constrained.yaml;; separate) CONFIG=configs/rebel_separate_constrained.yaml;; *) echo "unknown arm"; exit 1;; esac
test -f artifacts/rebel/train_prompts.jsonl || bash scripts/run_rebel_prepare.sh
python3 -m saferl.train_rebel --config "$CONFIG" --seed "$SEED" "$@"
