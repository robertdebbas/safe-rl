#!/usr/bin/env bash
# Evaluate one arm on ALL 1,960 held-out prompts in one GPU job (generation + Beaver scoring, single shard); `--with-base` first
# evaluates the untrained base policy in the same job (arm 0's job carries it). Merging and bootstraps are CPU-only (eval stage).
#   bash scripts/run_sixarm_eval.sh 0 --with-base
set -euo pipefail
cd "$(dirname "$0")/.."
K="${1:?arm index 0-5}"; shift || true
export PYTHONPATH=src CONFIG=configs/eval_sixarm.yaml
if [ "${1:-}" = "--with-base" ]; then bash scripts/run_eval_shard.sh base 0 1 0; fi
bash scripts/run_eval_shard.sh "sixarm_arm$K" 42 1 0
