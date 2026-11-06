#!/usr/bin/env python3
"""Compare the calibrated cost scales of the two constrained arms and summarize the d representativeness checks.
Report only: never changes d, lambda_0 or eta_lambda.
    python scripts/grpo_calibration_report.py [--artifacts artifacts]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--scale_ratio_flag", type=float, default=2.0, help="flag if std ratio between arms exceeds this")
    ap.add_argument("--repr_flag_std", type=float, default=0.25, help="flag if |train-sample mean - d| exceeds this many calibration std")
    args = ap.parse_args()
    art = Path(args.artifacts) / "grpo"
    cal = {a: json.load(open(art / f"calibration_{a}.json")) for a in ("shared", "separate") if (art / f"calibration_{a}.json").exists()}
    rep = {a: json.load(open(art / f"calibration_{a}_representativeness.json")) for a in ("shared", "separate")
           if (art / f"calibration_{a}_representativeness.json").exists()}
    out = {"calibration_scales": {}, "representativeness": {}, "flags": []}
    print("| arm | n | d = mean C | std C | p5 | median | p95 | min | max | mean R |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for a, c in cal.items():
        cc = c["cost"]
        out["calibration_scales"][a] = {"d": c["d"], **cc, "reward_mean": c["reward"]["mean"], "n": c["n_trajectories"]}
        print(f"| {a} | {c['n_trajectories']} | {c['d']:.4f} | {cc['std']:.4f} | {cc['p5']:.4f} | {cc['median']:.4f} | {cc['p95']:.4f} | {cc['min']:.4f} | {cc['max']:.4f} | {c['reward']['mean']:.4f} |")
    if len(cal) == 2:
        s1, s2 = cal["shared"]["cost"]["std"], cal["separate"]["cost"]["std"]
        ratio = max(s1, s2) / max(min(s1, s2), 1e-12)
        iqr = {a: cal[a]["cost"]["p95"] - cal[a]["cost"]["p5"] for a in cal}
        out["scale_comparison"] = {"std_ratio_max_over_min": ratio, "p5_p95_width": iqr}
        print(f"\nstd(C) ratio (max/min) = {ratio:.3f}; p5-p95 width shared = {iqr['shared']:.4f}, separate = {iqr['separate']:.4f}")
        if ratio > args.scale_ratio_flag:
            out["flags"].append(f"cost scales differ substantially (std ratio {ratio:.2f} > {args.scale_ratio_flag}); lambda_0/eta unchanged, review before launch")
    for a, r in rep.items():
        out["representativeness"][a] = {k: r[k] for k in ("d", "train_sample_mean_cost_minus_d", "diff_in_calibration_std_units", "diff_in_standard_errors")}
        out["representativeness"][a]["train_sample_cost"] = r["train_sample"]["cost"]
        print(f"\n{a}: d = {r['d']:.4f}; train-sample (n={r['train_sample']['n_trajectories']}) mean C = {r['train_sample']['cost']['mean']:.4f}; "
              f"diff = {r['train_sample_mean_cost_minus_d']:+.4f} ({r['diff_in_calibration_std_units']:+.2f} calibration std, {r['diff_in_standard_errors']:+.2f} SE)")
        if abs(r["diff_in_calibration_std_units"] or 0) > args.repr_flag_std:
            out["flags"].append(f"{a}: d may not be representative of training prompts (|diff| = {abs(r['diff_in_calibration_std_units']):.2f} std > {args.repr_flag_std}); flag before launch")
    print("\nflags:", out["flags"] or "none")
    json.dump(out, open(art / "calibration_report.json", "w"), indent=2)


if __name__ == "__main__":
    main()
