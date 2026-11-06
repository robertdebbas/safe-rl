"""Aggregate evaluation results across training seeds: mean and standard deviation (sample
std, ddof=1) of every subgroup metric on every evaluated split, plus the conflict diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .metrics import GROUP_ORDER, STAT_KEYS
from .utils import read_json, write_json

METRIC_KEYS = STAT_KEYS


def _mean_std(values: List[float]) -> Dict[str, Any]:
    arr = np.array(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0, "n_seeds": int(arr.size), "values": [float(v) for v in arr]}


def aggregate(run_dirs: List[Path]) -> Dict[str, Any]:
    evals = {}
    trains = {}
    for rd in run_dirs:
        ev = read_json(rd / "eval" / "eval_summary.json")
        tr = read_json(rd / "train_summary.json") if (rd / "train_summary.json").exists() else {}
        evals[rd.name] = ev
        trains[rd.name] = tr
    splits = sorted(set.intersection(*[set(ev["splits"]) for ev in evals.values()]))
    out: Dict[str, Any] = {"runs": [str(r) for r in run_dirs], "n_seeds": len(run_dirs), "splits": {}}
    for s in splits:
        agg: Dict[str, Any] = {"groups": {}, "combined_loss": _mean_std([ev["splits"][s]["combined_loss"] for ev in evals.values()])}
        for g in GROUP_ORDER:
            n_vals = [ev["splits"][s]["groups"][g]["n"] for ev in evals.values()]
            agg["groups"][g] = {"n": n_vals[0], "n_consistent": len(set(n_vals)) == 1}
            if n_vals[0] == 0:
                continue
            for k in METRIC_KEYS:
                agg["groups"][g][k] = _mean_std([ev["splits"][s]["groups"][g][k] for ev in evals.values()])
            agg["groups"][g]["n_prompts"] = evals[next(iter(evals))]["splits"][s]["groups"][g]["n_prompts"]
            agg["groups"][g]["prompt_averaged"] = {k: _mean_std([ev["splits"][s]["groups"][g]["prompt_averaged"][k] for ev in evals.values()]) for k in METRIC_KEYS}
        d = [ev["splits"][s]["conflict_diagnostic"] for ev in evals.values()]
        agg["conflict_diagnostic"] = {
            "n_conflict_pairs": d[0]["n_conflict_pairs"],
            "ties_frac": _mean_std([x["ties"]["frac"] or 0.0 for x in d]),
            "prompt_averaged": {k: _mean_std([x["prompt_averaged"][k] or 0.0 for x in d]) for k in
                                ("favors_helpfulness_preferred_frac", "favors_safety_preferred_frac", "mean_margin_toward_helpfulness_preferred")},
            "favors_helpfulness_preferred_frac": _mean_std([x["favors_helpfulness_preferred"]["frac"] or 0.0 for x in d]),
            "favors_safety_preferred_frac": _mean_std([x["favors_safety_preferred"]["frac"] or 0.0 for x in d]),
            "mean_margin_toward_helpfulness_preferred": _mean_std([x["mean_margin_toward_helpfulness_preferred"] or 0.0 for x in d]),
        }
        out["splits"][s] = agg
    out["training"] = {name: {"best_step": tr.get("best_step"), "best_combined_val_loss": tr.get("best_combined_val_loss"),
                              "total_steps": tr.get("total_steps"), "seed": tr.get("seed")} for name, tr in trains.items()}
    return out


def render(agg: Dict[str, Any]) -> str:
    md = [f"# Aggregate over {agg['n_seeds']} seeds\n", "runs: " + ", ".join(agg["runs"]) + "\n"]
    md.append("## Checkpoint selection (validation combined loss)\n")
    md.append("| run | seed | best step | total steps | best combined val loss |\n|---|---|---|---|---|")
    for name, t in agg["training"].items():
        md.append(f"| {name} | {t['seed']} | {t['best_step']} | {t['total_steps']} | {t['best_combined_val_loss']:.4f} |" if t.get("best_combined_val_loss") is not None else f"| {name} | - | - | - | - |")
    md.append("")
    for s, a in agg["splits"].items():
        md.append(f"## {s}\n")
        for pa, label in ((False, "Pair-averaged"), (True, "Prompt-averaged")):
            md.append(f"**{label}**\n")
            md.append("| group | n | n_prompts | loss | pref_acc | pref_acc_tie_adj | margin>0 | margin=0 | margin<0 | mean_margin |\n|---|---|---|---|---|---|---|---|---|---|")
            for g in GROUP_ORDER:
                x = a["groups"][g]
                if x["n"] == 0:
                    continue
                v = x["prompt_averaged"] if pa else x
                f = lambda k: f"{v[k]['mean']:.4f} ± {v[k]['std']:.4f}"  # noqa: E731
                md.append(f"| {g} | {x['n']} | {x['n_prompts']} | {f('loss')} | {f('pref_acc')} | {f('pref_acc_tie_adjusted')} | "
                          f"{f('frac_margin_positive')} | {f('frac_margin_zero')} | {f('frac_margin_negative')} | {f('mean_margin')} |")
            md.append("")
        c = a["combined_loss"]
        md.append(f"\ncombined loss (0.5 H + 0.5 S): {c['mean']:.4f} ± {c['std']:.4f}")
        d = a["conflict_diagnostic"]
        md.append(f"\nconflict pairs (n={d['n_conflict_pairs']}): favors helpfulness-preferred {100*d['favors_helpfulness_preferred_frac']['mean']:.1f}% ± {100*d['favors_helpfulness_preferred_frac']['std']:.1f}; "
                  f"favors safety-preferred {100*d['favors_safety_preferred_frac']['mean']:.1f}% ± {100*d['favors_safety_preferred_frac']['std']:.1f}; "
                  f"ties {100*d['ties_frac']['mean']:.1f}% ± {100*d['ties_frac']['std']:.1f}; "
                  f"mean margin toward helpfulness-preferred {d['mean_margin_toward_helpfulness_preferred']['mean']:.4f} ± {d['mean_margin_toward_helpfulness_preferred']['std']:.4f}; "
                  f"prompt-averaged: helpfulness {100*d['prompt_averaged']['favors_helpfulness_preferred_frac']['mean']:.1f}% ± {100*d['prompt_averaged']['favors_helpfulness_preferred_frac']['std']:.1f}, "
                  f"safety {100*d['prompt_averaged']['favors_safety_preferred_frac']['mean']:.1f}% ± {100*d['prompt_averaged']['favors_safety_preferred_frac']['std']:.1f}\n")
    return "\n".join(md)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (one per seed)")
    ap.add_argument("--out", default="runs/aggregate")
    args = ap.parse_args(argv)
    agg = aggregate([Path(r) for r in args.runs])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", agg)
    (out / "summary.md").write_text(render(agg), encoding="utf-8")
    print(render(agg))


if __name__ == "__main__":
    main()
