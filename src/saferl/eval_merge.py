"""Merge evaluation shards, verify coverage, and compute the final table with seed-stratified bootstrap CIs.

Per method and metric (helpfulness = reward_score, harmfulness = cost_score):
  seed mean_s        = mean over the manifest prompts of seed s
  overall point      = mean over seeds of mean_s (equal seed weight)
  bootstrap (2,000)  = for each replicate: resample the prompts WITH replacement WITHIN each seed
                       separately, take each seed's mean, average the seed means; CI = 2.5th / 97.5th pct.
Seeds are never pooled into one flat sample.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .eval_common import load_eval_prompts, method_checkpoint, run_dir
from .utils import get_logger, load_config, read_json, read_jsonl, write_json, write_jsonl

log = get_logger("saferl.eval_merge")
SHARD_RE = re.compile(r"^(responses|scores)\.shard(\d+)of(\d+)\.jsonl$")


def merge_shards(d: Path, kind: str, manifest_hashes: List[str]) -> Dict[str, Any]:
    """Concatenate all shards of ``kind`` in ``d`` after verifying they form one complete, duplicate-free
    cover of the manifest (same num_shards everywhere, all shard ids present)."""
    files = sorted(p for p in d.glob(f"{kind}.shard*of*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no {kind} shards in {d}")
    parsed = [(p, SHARD_RE.match(p.name)) for p in files]
    totals = {int(m.group(3)) for _, m in parsed}
    if len(totals) != 1:
        raise ValueError(f"{d}: shards from different layouts present: num_shards={sorted(totals)}")
    n_shards = totals.pop()
    ids = sorted(int(m.group(2)) for _, m in parsed)
    if ids != list(range(n_shards)):
        raise ValueError(f"{d}: missing shards: have {ids}, expected 0..{n_shards - 1}")
    rows: List[Dict[str, Any]] = []
    for p, _ in parsed:
        rows.extend(read_jsonl(p))
    hashes = [r["prompt_hash"] for r in rows]
    dup = {h for h in hashes if hashes.count(h) > 1} if len(hashes) != len(set(hashes)) else set()
    missing = set(manifest_hashes) - set(hashes)
    extra = set(hashes) - set(manifest_hashes)
    if dup or missing or extra:
        raise ValueError(f"{d}/{kind}: duplicates={len(dup)} missing={len(missing)} extra={len(extra)}")
    rows.sort(key=lambda r: r["manifest_index"])
    merged = d / f"{kind}.jsonl"
    write_jsonl(merged, rows)
    return {"rows": rows, "n_shards": n_shards, "file": str(merged), "n": len(rows)}


def bootstrap_ci(per_seed: Dict[Any, np.ndarray], n_replicates: int, rng_seed: int, q_low: float = 2.5, q_high: float = 97.5) -> Dict[str, Any]:
    """Seed-stratified bootstrap of the equal-weight mean of seed means."""
    rng = np.random.default_rng(rng_seed)
    seeds = sorted(per_seed)
    boot = np.zeros(n_replicates, dtype=np.float64)
    for s in seeds:                                   # resample prompts within THIS seed only
        x = np.asarray(per_seed[s], dtype=np.float64)
        idx = rng.integers(0, len(x), size=(n_replicates, len(x)))
        boot += x[idx].mean(axis=1)
    boot /= len(seeds)                                # equal weight per seed
    seed_means = {str(s): float(np.mean(per_seed[s])) for s in seeds}
    return {"point": float(np.mean(list(seed_means.values()))), "ci_low": float(np.percentile(boot, q_low)),
            "ci_high": float(np.percentile(boot, q_high)), "seed_means": seed_means, "n_seeds": len(seeds),
            "n_per_seed": {str(s): int(len(per_seed[s])) for s in seeds}, "n_replicates": int(n_replicates), "bootstrap_seed": int(rng_seed)}


def summarize_method(cfg: Dict[str, Any], method: str, manifest_hashes: List[str]) -> Optional[Dict[str, Any]]:
    mcfg = cfg["methods"][method]
    per_seed = {"reward": {}, "cost": {}}
    meta = {"label": mcfg["label"], "checkpoints": {}, "n_shards": {}, "generation_truncation_rate": {}, "evaluator_input_truncated": {}}
    for seed in mcfg["seeds"]:
        d = run_dir(cfg, method, seed)
        if not list(d.glob("scores.shard*of*.jsonl")):
            log.warning("%s seed %s: no score shards in %s; method skipped", method, seed, d)
            return None
        sc = merge_shards(d, "scores", manifest_hashes)
        try:
            merge_shards(d, "responses", manifest_hashes)
        except FileNotFoundError:
            pass
        rows = sc["rows"]
        per_seed["reward"][seed] = np.array([r["reward_score"] for r in rows])
        per_seed["cost"][seed] = np.array([r["cost_score"] for r in rows])
        meta["checkpoints"][str(seed)] = method_checkpoint(mcfg, seed)
        meta["n_shards"][str(seed)] = sc["n_shards"]
        meta["generation_truncation_rate"][str(seed)] = float(np.mean([r["truncated"] for r in rows]))
        meta["evaluator_input_truncated"][str(seed)] = int(sum(r["evaluator_input_truncated"] for r in rows))
    b = cfg["bootstrap"]
    return {"method": method, **meta,
            "helpfulness": bootstrap_ci(per_seed["reward"], int(b["n_replicates"]), int(b["seed"]), *b.get("percentiles", [2.5, 97.5])),
            "harmfulness": bootstrap_ci(per_seed["cost"], int(b["n_replicates"]), int(b["seed"]) + 1, *b.get("percentiles", [2.5, 97.5]))}


def render_table(results: List[Dict[str, Any]]) -> str:
    lines = ["| Method | Avg Helpfulness ↑ | Avg Harmfulness ↓ | seeds | n/seed |", "|---|---:|---:|---|---|"]
    for r in results:
        h, c = r["helpfulness"], r["harmfulness"]
        lines.append(f"| {r['label']} | {h['point']:.3f} [{h['ci_low']:.3f}, {h['ci_high']:.3f}] | "
                     f"{c['point']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}] | {h['n_seeds']} | {list(h['n_per_seed'].values())[0]} |")
    return "\n".join(lines)


def merge_and_summarize(cfg: Dict[str, Any], methods: Optional[List[str]] = None, seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    """``seeds`` restricts trained methods to a subset of policy seeds (e.g. a preliminary seed-42 run); the
    single-response-set base (seeds [0]) is unaffected. The summary records which seeds were used."""
    if seeds:
        cfg = {**cfg, "methods": {m: ({**v, "seeds": [s for s in v["seeds"] if s in seeds]} if v["seeds"] != [0] else v)
                                  for m, v in cfg["methods"].items()}}
    prompts = load_eval_prompts(cfg["artifacts_dir"])
    hashes = [p["prompt_hash"] for p in prompts]
    methods = methods or list(cfg["methods"])
    results = [r for r in (summarize_method(cfg, m, hashes) for m in methods) if r is not None]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_info = read_json(Path(cfg["artifacts_dir"]) / "eval" / "manifest_info.json")
    summary = {"n_prompts": len(prompts), "manifest": manifest_info, "eval_seed": cfg["eval_seed"], "generation": cfg["generation"],
               "policy_seeds_used": {m: v["seeds"] for m, v in cfg["methods"].items()}, "preliminary_seed_subset": bool(seeds),
               "evaluators": cfg["evaluators"], "bootstrap": cfg["bootstrap"], "results": results}
    write_json(out_dir / "summary.json", summary)
    table = render_table(results)
    (out_dir / "summary.md").write_text(
        f"# Final comparison ({len(prompts)} evaluation prompts, {cfg['bootstrap']['n_replicates']} seed-stratified bootstrap replicates, 95% CI)\n\n"
        f"Evaluators: {cfg['evaluators']['reward']['model']} (helpfulness, higher better), {cfg['evaluators']['cost']['model']} (harmfulness, lower better).\n\n"
        + table + "\n", encoding="utf-8")
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "label", "metric", "point", "ci_low", "ci_high", "n_seeds", "seed", "seed_mean", "n_prompts", "checkpoint"])
        for r in results:
            for metric in ("helpfulness", "harmfulness"):
                m = r[metric]
                for s, v in m["seed_means"].items():
                    w.writerow([r["method"], r["label"], metric, m["point"], m["ci_low"], m["ci_high"], m["n_seeds"], s, v, m["n_per_seed"][s], r["checkpoints"][s]])
    print(table)
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="restrict trained methods to these policy seeds")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    merge_and_summarize(load_config(args.config, args.set), args.methods, args.seeds)


if __name__ == "__main__":
    main()
