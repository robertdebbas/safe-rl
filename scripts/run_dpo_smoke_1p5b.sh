#!/usr/bin/env bash
# GPU engineering smoke test with the SCIENTIFIC model (Qwen2.5-1.5B-Instruct): tiny data subset,
# 4 optimizer steps, train + checkpoint selection + eval + summary. Run this on the GPU box BEFORE any
# full scientific seed. Numbers are not scientific.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export WORLD_SIZE="${WORLD_SIZE:-1}"
ART=artifacts/smoke_1p5b
test -f "$ART/length_limits.json" || python3 scripts/prepare_data.py --config configs/data_smoke.yaml \
    --set tokenizer=Qwen/Qwen2.5-1.5B-Instruct out_dir="$ART"
OVR=(model=Qwen/Qwen2.5-1.5B-Instruct artifacts_dir="$ART" ref_cache_dir="$ART/ref_logprobs" run_name=smoke_qwen2.5-1.5b
     per_device_batch_size=4 global_batch_size=8 eval_batch_size=8)
python3 -m saferl.train_dpo --config configs/dpo_smoke.yaml --seed 42 --set "${OVR[@]}" "$@"
python3 -m saferl.eval_dpo  --config configs/dpo_smoke.yaml --run_dir runs/smoke_qwen2.5-1.5b_seed42 --set "${OVR[@]}" "$@"
echo "--- 1.5B smoke eval summary ---"; cat runs/smoke_qwen2.5-1.5b_seed42/eval/eval_summary.md
