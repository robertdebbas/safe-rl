#!/usr/bin/env python3
"""Post-run diagnostics for GRPO runs: budget, monitor history, lambda/cost/reward/length traces, eval response stats + samples."""
import json, os, sys
import numpy as np

arms = sys.argv[1:] or ["scalar", "shared_constrained", "separate_constrained"]
for a in arms:
    rd = f"runs/grpo_{a}_qwen2.5-1.5b_seed42"
    if not os.path.exists(f"{rd}/train_summary.json"):
        print(a, "missing"); continue
    t = json.load(open(f"{rd}/train_summary.json"))
    print(f"=== {a}: prompts {t['n_prompts']} traj {t['n_trajectories']} gen tokens {t['n_generated_tokens']:,} steps {t['opt_step']} wall {t['wallclock_s']/3600:.2f}h "
          f"trunc {t['truncation_rate']:.3f} zero-var groups {t['n_zero_variance_groups']}/{t['n_groups']} d {t['d']} lambda_final {t['final_lambda']}")
    hist = json.load(open(f"{rd}/monitor_history.json"))
    keys = [k for k in ("tag", "kl_mean", "completion_len_mean", "truncation_rate", "scalar_reward_mean", "reward_mean", "cost_mean", "cost_minus_d", "lambda") if k in hist[0]]
    print("  monitor:", " | ".join(keys))
    for m in hist:
        print("   ", " | ".join(f"{m[k]:.3f}" if isinstance(m[k], float) else str(m[k]) for k in keys))
    rows = [json.loads(l) for l in open(f"{rd}/rollout_batches.jsonl")]
    idx = sorted(set([0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, len(rows) - 1]))
    idx = [i for i in idx if i < len(rows)]
    ck = ["batch_index", "completion_len_mean", "truncation_rate", "zero_variance_groups", "kl_mean", "clip_fraction", "adv_abs_mean"] + \
         (["lambda", "cost_batch_mean", "cost_ema", "reward_mean", "cost_std"] if "lambda" in rows[0] else ["scalar_reward_mean"])
    print("  rollout trace:", " | ".join(ck))
    for i in idx:
        r = rows[i]
        print("   ", " | ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in ck))
    ev = f"runs/eval/grpo_{a.replace('_constrained','')}/seed42/responses.jsonl"
    if os.path.exists(ev):
        resp = [json.loads(l) for l in open(ev)]
        L = np.array([x["n_completion_tokens"] for x in resp])
        empty = sum(1 for x in resp if len(x["response"].strip()) == 0)
        uniq = len({x["response"][:60] for x in resp})
        print(f"  eval responses: n {len(resp)} len mean {L.mean():.1f} p5 {np.percentile(L,5):.0f} p50 {np.percentile(L,50):.0f} p95 {np.percentile(L,95):.0f} "
              f"empty {empty} distinct-first-60-chars {uniq} truncated {np.mean([x['truncated'] for x in resp]):.3f}")
        for x in resp[:3]:
            print("    PROMPT:", x["prompt"][:100].replace("\n", " "))
            print("    RESP  :", x["response"][:220].replace("\n", " "))
