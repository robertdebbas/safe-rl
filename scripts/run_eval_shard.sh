#!/usr/bin/env bash
# Generate + score one (method, policy seed, shard): a normal single-GPU job.
#   bash scripts/run_eval_shard.sh dpo 42 4 0        # method seed num_shards shard_id
#   bash scripts/run_eval_shard.sh base 0 1 0        # one-shard pilot of the base model
set -euo pipefail
cd "$(dirname "$0")/.."
METHOD="${1:?method}"; SEED="${2:?seed}"; N="${3:-1}"; I="${4:-0}"; shift 4 || true
export PYTHONPATH=src
CONFIG="${CONFIG:-configs/eval.yaml}"
test -f artifacts/eval/eval_prompts.jsonl || bash scripts/run_eval_prepare.sh
python3 -m saferl.eval_generate --config "$CONFIG" --method "$METHOD" --seed "$SEED" --num-shards "$N" --shard-id "$I" "$@"
python3 -m saferl.eval_score    --config "$CONFIG" --method "$METHOD" --seed "$SEED" --num-shards "$N" --shard-id "$I" "$@"
