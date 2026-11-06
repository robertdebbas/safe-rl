#!/usr/bin/env bash
# Verify shard coverage, concatenate, compute seed means + seed-stratified bootstrap CIs, write the final table.
#   bash scripts/run_eval_merge.sh                 -> runs/eval/summary.{md,json,csv}
#   bash scripts/run_eval_merge.sh base dpo        -> only these methods
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
CONFIG="${CONFIG:-configs/eval.yaml}"
if [ $# -gt 0 ]; then python3 -m saferl.eval_merge --config "$CONFIG" --methods "$@"; else python3 -m saferl.eval_merge --config "$CONFIG"; fi
