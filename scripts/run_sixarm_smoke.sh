#!/usr/bin/env bash
# 1.5B GPU smoke of one six-arm config: 3 rollout batches with the real scorers and calibration, cadences compressed; engineering only.
#   bash scripts/run_sixarm_smoke.sh arm0_rebel_separate_d1 | arm3_grpo_separate_d1 | ...
set -euo pipefail
cd "$(dirname "$0")/.."
A="${1:?arm config name}"
export PYTHONPATH=src WORLD_SIZE=1
CFG="configs/sixarm/$A.yaml"
case "$A" in *rebel*) MOD=saferl.train_rebel;; *) MOD=saferl.train_grpo;; esac
rm -rf "runs/sixarm_smoke_${A}_seed42"
python3 -m "$MOD" --config "$CFG" --seed 42 --set run_name="sixarm_smoke_$A" max_rollout_batches=3 checkpoint_at_batches="[2,3]" eval_every_batches=2 \
    resume_every_batches=2 dump_every_batches=1 monitor_responses_per_prompt=1 monitor_gen_batch=32 log_every_batches=1
python3 - "$A" <<'PY'
import json, sys
d = f"runs/sixarm_smoke_{sys.argv[1]}_seed42"
s = json.load(open(f"{d}/train_summary.json"))
print({k: s.get(k) for k in ("n_prompts", "n_trajectories", "n_pairs", "opt_step", "truncation_rate", "wallclock_s", "memory", "final_lambda", "d", "d_std")})
rows = [json.loads(l) for l in open(f"{d}/rollout_batches.jsonl")]
print("per-batch seconds:", [round(r["t_batch_s"]) for r in rows], "generate:", [round(r["t_generate_s"]) for r in rows], "train:", [round(r["t_train_s"]) for r in rows])
print("lambda:", [r.get("lambda") for r in rows], "cost:", [round(r.get("cost_batch_mean", r.get("cost_hat", float('nan'))), 3) for r in rows])
import os; print("checkpoints:", sorted(os.listdir(f"{d}/checkpoints")))
PY
