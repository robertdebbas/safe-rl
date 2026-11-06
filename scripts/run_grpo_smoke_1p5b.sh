#!/usr/bin/env bash
# 1.5B GPU smoke of one GRPO arm: 3 rollout batches with the real seed-42 scorers and the real calibrated d, tiny prompt counts.
#   bash scripts/run_grpo_smoke_1p5b.sh scalar|shared|separate
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?arm}"
export PYTHONPATH=src
case "$ARM" in scalar) CONFIG=configs/grpo_scalar.yaml;; shared) CONFIG=configs/grpo_shared_constrained.yaml;; separate) CONFIG=configs/grpo_separate_constrained.yaml;; esac
test -f artifacts/grpo/train_prompts.jsonl || bash scripts/run_grpo_prepare.sh
rm -rf "runs/grpo_smoke1p5b_${ARM}_seed42"
python3 -m saferl.train_grpo --config "$CONFIG" --seed 42 --set run_name="grpo_smoke1p5b_${ARM}" max_rollout_batches=3 eval_every_batches=2 \
    resume_every_batches=2 monitor_responses_per_prompt=1 monitor_gen_batch=32 log_every_batches=1
python3 - "$ARM" <<'PY'
import json, sys
s = json.load(open(f"runs/grpo_smoke1p5b_{sys.argv[1]}_seed42/train_summary.json"))
print({k: s[k] for k in ("n_prompts", "n_trajectories", "n_generated_tokens", "opt_step", "truncation_rate", "wallclock_s", "memory", "final_lambda", "d")})
rows = [json.loads(l) for l in open(f"runs/grpo_smoke1p5b_{sys.argv[1]}_seed42/rollout_batches.jsonl")]
print("per-batch seconds:", [round(r["t_batch_s"]) for r in rows], "generate:", [round(r["t_generate_s"]) for r in rows], "train:", [round(r["t_train_s"]) for r in rows])
print("completion len mean / p95:", [(round(r["completion_len_mean"]), round(r["completion_len_p95"])) for r in rows], "truncation:", [round(r["truncation_rate"], 3) for r in rows])
PY
