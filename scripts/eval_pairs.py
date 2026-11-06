"""Paired, prompt-level bootstrap comparisons between evaluated methods (six-arm study) plus the six-row summary table.

For each pair (A, B) in the eval config's ``paired_comparisons``: on the identical 1,960 prompts (same per-prompt sampling seeds),
diff_i = score_A(i) - score_B(i); the point estimate is mean(diff) and the 95% CI comes from 2,000 bootstrap replicates that
resample PROMPTS with replacement (the same resample applied to both arms), separately for Beaver helpfulness and harmfulness.
Reads runs/eval/<method>/seed<seed>/scores.jsonl written by eval_score (one policy seed per arm: the intervals quantify
evaluation-prompt uncertainty only, not training-seed variation). Writes <out_dir>/sixarm_pairs.json and .md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saferl.eval_common import run_dir  # noqa: E402
from saferl.utils import load_config, read_jsonl  # noqa: E402


def load_scores(cfg: Dict, method: str) -> Dict[str, Dict[str, float]]:
    seed = cfg["methods"][method]["seeds"][0]
    d = run_dir(cfg, method, seed)
    f = d / "scores.jsonl"
    if not f.exists():
        shards = sorted(d.glob("scores.shard*of*.jsonl"))
        if not shards:
            raise FileNotFoundError(f"no scores for {method} in {d}")
        rows = [r for p in shards for r in read_jsonl(p)]
    else:
        rows = read_jsonl(f)
    return {r["prompt_hash"]: {"R": float(r["reward_score"]), "C": float(r["cost_score"])} for r in rows}


def paired(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]], n_rep: int, seed: int) -> Dict[str, Dict]:
    common = sorted(set(a) & set(b))
    if len(common) != len(a) or len(common) != len(b):
        raise ValueError(f"prompt sets differ: {len(a)} vs {len(b)} vs common {len(common)}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), (n_rep, len(common)))
    out = {}
    for key, label in (("R", "helpfulness"), ("C", "harmfulness")):
        diff = np.array([a[h][key] - b[h][key] for h in common])
        boot = diff[idx].mean(axis=1)
        out[label] = {"point": float(diff.mean()), "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
                      "frac_prompts_a_higher": float(np.mean(diff > 0)), "n_prompts": len(common)}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_sixarm.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    b = cfg["bootstrap"]
    pairs: List[List[str]] = cfg.get("paired_comparisons", [])
    scores = {}
    results = []
    for a_m, b_m in pairs:
        for m in (a_m, b_m):
            if m not in scores:
                scores[m] = load_scores(cfg, m)
        res = paired(scores[a_m], scores[b_m], int(b["n_replicates"]), int(b["seed"]) + 7)
        results.append({"a": a_m, "b": b_m, "label_a": cfg["methods"][a_m]["label"], "label_b": cfg["methods"][b_m]["label"], **res})
    out_dir = Path(cfg["out_dir"])
    lines = ["| Comparison (A − B) | Δ helpfulness [95% CI] | Δ harmfulness [95% CI] | n prompts |", "|---|---:|---:|---|"]
    for r in results:
        h, c = r["helpfulness"], r["harmfulness"]
        lines.append(f"| {r['label_a']} − {r['label_b']} | {h['point']:+.3f} [{h['ci_low']:+.3f}, {h['ci_high']:+.3f}] | "
                     f"{c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {h['n_prompts']} |")
    md = ("# Paired prompt-level bootstrap comparisons (same prompts and sampling seeds; one policy seed per arm, so the intervals\n"
          "# reflect evaluation-prompt uncertainty only, not training-seed variation)\n\n" + "\n".join(lines) + "\n")
    (out_dir / "sixarm_pairs.md").write_text(md, encoding="utf-8")
    (out_dir / "sixarm_pairs.json").write_text(json.dumps({"bootstrap": b, "pairs": results}, indent=2))
    print(md)


if __name__ == "__main__":
    main()
