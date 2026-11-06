#!/usr/bin/env python3
"""Live progress of training runs from their flushed metrics files (Slurm .out files are block-buffered).
    python scripts/slurm/progress.py [runs/...]   (default: all runs/*/metrics.jsonl and GRPO rollout logs)"""
import glob
import json
import os
import sys
import time

paths = sys.argv[1:] or sorted(glob.glob("runs/*/metrics.jsonl")) + sorted(glob.glob("runs/*/rollout_batches.jsonl"))
for p in paths:
    run = p.split("/")[1]
    try:
        rows = [json.loads(l) for l in open(p)]
    except Exception as e:
        print(f"{run}: unreadable ({e})"); continue
    age = (time.time() - os.path.getmtime(p)) / 60
    if p.endswith("rollout_batches.jsonl"):
        if not rows:
            print(f"{run}: no rollout batches yet"); continue
        r = rows[-1]
        print(f"{run}: rollout batch {r['batch_index']+1}, {r['t_batch_s']:.0f}s/batch (gen {r['t_generate_s']:.0f}s train {r['t_train_s']:.0f}s), "
              f"trunc {r['truncation_rate']:.3f}, len {r['completion_len_mean']:.0f}, kl {r['kl_mean']:.4f}"
              + (f", lambda {r['lambda']:.3f} cost {r['cost_batch_mean']:.3f} d {r['d']:.3f}" if "lambda" in r else f", R {r.get('scalar_reward_mean', float('nan')):.3f}")
              + f"  [updated {age:.0f} min ago]")
        continue
    tr = [r for r in rows if r.get("type") == "train"]; va = [r for r in rows if r.get("type") == "val"]
    if tr:
        t = [r["elapsed_s"] for r in tr]; per = (t[-1] - t[0]) / max(1, len(t) - 1)
        tot = None
        cfg = f"runs/{run}/config_resolved.json"
        print(f"{run}: step {tr[-1]['step']}, {t[-1]/60:.1f} min elapsed, {per:.1f} s/step, loss {tr[-1]['loss']:.4f}, val passes {len(va)}"
              + (f", last val combined {va[-1]['combined_loss']:.4f}" if va else "") + f"  [updated {age:.0f} min ago]")
    else:
        print(f"{run}: {len(va)} val passes, no train steps yet  [updated {age:.0f} min ago]")
