#!/usr/bin/env bash
# Short GPU smoke of one REBEL arm with the REAL 1.5B policy, real seed-42 scorers and the real calibration: 3 rollout batches,
# monitoring/dump/checkpoint cadence compressed. Engineering only (no scientific setting changes). ~10-15 min on one A100.
#   bash scripts/run_rebel_smoke_1p5b.sh scalar|shared|separate
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?arm}"
export PYTHONPATH=src
case "$ARM" in scalar) CONFIG=configs/rebel_scalar.yaml;; shared) CONFIG=configs/rebel_shared_constrained.yaml;; separate) CONFIG=configs/rebel_separate_constrained.yaml;; esac
test -f artifacts/rebel/train_prompts.jsonl || bash scripts/run_rebel_prepare.sh
rm -rf "runs/rebel_smoke1p5b_${ARM}_seed42"
python3 -m saferl.train_rebel --config "$CONFIG" --seed 42 --set run_name="rebel_smoke1p5b_${ARM}" max_rollout_batches=3 eval_every_batches=2 \
    resume_every_batches=2 checkpoint_every_batches=2 dump_every_batches=1 monitor_responses_per_prompt=1 monitor_gen_batch=32 log_every_batches=1
python3 - "$ARM" <<'PY'
import json, sys
s = json.load(open(f"runs/rebel_smoke1p5b_{sys.argv[1]}_seed42/train_summary.json"))
print({k: s[k] for k in ("n_prompts", "n_trajectories", "n_pairs", "n_generated_tokens", "opt_step", "truncation_rate", "wallclock_s", "memory", "final_lambda", "d_std")})
rows = [json.loads(l) for l in open(f"runs/rebel_smoke1p5b_{sys.argv[1]}_seed42/rollout_batches.jsonl")]
print("per-batch seconds:", [round(r["t_batch_s"]) for r in rows], "generate:", [round(r["t_generate_s"]) for r in rows], "train:", [round(r["t_train_s"]) for r in rows])
print("loss (at theta_t -> minibatch mean -> post-update):", [(round(r["loss_at_theta_t"], 3), round(r["loss"], 3), round(r.get("post_loss", float('nan')), 3)) for r in rows])
print("post-update pred/target corr:", [round(r.get("post_pred_target_corr", float('nan')), 3) for r in rows], "KL(pi_t||pi_t+1) token:", [r.get("kl_prev", {}).get("kl_k3_token_mean") for r in rows])
print("len mean / trunc:", [(round(r["completion_len_mean"]), round(r["truncation_rate"], 3)) for r in rows], "grad norms:", [[round(g, 2) for g in r["grad_norm"]] for r in rows])
PY
