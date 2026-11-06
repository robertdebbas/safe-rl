"""Aggregate reward/cost-model evaluations across seeds: mean ± sample std (ddof=1) of every numeric
leaf in the per-split metric trees, plus parameter counts and training compute from the runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .rm_metrics import GROUP_ORDER
from .utils import read_json, write_json


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def mean_std_tree(trees: List[Any]) -> Any:
    """Recursively replace numeric leaves by {mean, std, n_seeds, values}; non-numeric leaves must agree."""
    first = trees[0]
    if isinstance(first, dict):
        return {k: mean_std_tree([t[k] for t in trees]) for k in first if all(isinstance(t, dict) and k in t for t in trees)}
    if _is_num(first) or first is None:
        vals = [t for t in trees if _is_num(t) and not (isinstance(t, float) and np.isnan(t))]
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0, "n_seeds": int(arr.size), "values": vals}
    return first


def aggregate(run_dirs: List[Path]) -> Dict[str, Any]:
    evals = [read_json(rd / "eval" / "eval_summary.json") for rd in run_dirs]
    trains = [read_json(rd / "train_summary.json") if (rd / "train_summary.json").exists() else {} for rd in run_dirs]
    splits = sorted(set.intersection(*[set(e["splits"]) for e in evals]))
    out = {"runs": [str(r) for r in run_dirs], "n_seeds": len(run_dirs), "arch": evals[0].get("arch"),
           "parameters": evals[0].get("parameters"), "splits": {s: mean_std_tree([e["splits"][s] for e in evals]) for s in splits},
           "training": [{"run": str(rd), "seed": t.get("seed"), "best_step": t.get("best_step"), "total_steps": t.get("total_steps"),
                         "best_combined_val_loss": t.get("best_combined_val_loss"), "elapsed_s": t.get("elapsed_s"),
                         "approx_train_flops": t.get("approx_train_flops"), "train_tokens_per_epoch": t.get("train_tokens_per_epoch")}
                        for rd, t in zip(run_dirs, trains)]}
    return out


def _f(x) -> str:
    return "-" if not isinstance(x, dict) or x.get("mean") is None else f"{x['mean']:.4f} ± {x['std']:.4f}"


def render(agg: Dict[str, Any]) -> str:
    md = [f"# Aggregate ({agg['arch']}) over {agg['n_seeds']} seeds\n", f"parameters: {agg['parameters']}\n",
          "| run | seed | best step | total steps | best combined val loss | wallclock s | approx FLOPs |\n|---|---|---|---|---|---|---|"]
    for t in agg["training"]:
        bl = t["best_combined_val_loss"]
        md.append(f"| {Path(t['run']).name} | {t['seed']} | {t['best_step']} | {t['total_steps']} | {bl:.4f} | {t['elapsed_s']:.0f} | {t['approx_train_flops']:.3e} |"
                  if bl is not None else f"| {Path(t['run']).name} | - | - | - | - | - | - |")
    md.append("")
    keys = ("loss", "acc", "acc_tie_adjusted", "frac_margin_zero", "mean_margin", "mean_cost_safe", "mean_cost_unsafe", "cost_gap_unsafe_minus_safe", "class_loss", "class_acc")
    for s, a in agg["splits"].items():
        md.append(f"## {s}\n")
        for pa, label in ((False, "Pair-averaged"), (True, "Prompt-averaged")):
            md.append(f"**{label}**\n")
            md.append("| group | n | " + " | ".join(keys) + " |\n|---|---|" + "|".join("---" for _ in keys) + "|")
            for g in GROUP_ORDER:
                x = a["groups"].get(g)
                if not x or not x.get("n") or x["n"]["mean"] == 0:
                    continue
                v = x["prompt_averaged"] if pa else x
                md.append(f"| {g} | {int(x['n']['mean'])} | " + " | ".join(_f(v.get(k)) for k in keys) + " |")
            md.append("")
        md.append(f"combined loss: {_f(a['combined_loss'])}\n")
    return "\n".join(md)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    agg = aggregate([Path(r) for r in args.runs])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", agg)
    (out / "summary.md").write_text(render(agg), encoding="utf-8")
    print(render(agg))


if __name__ == "__main__":
    main()
