"""Six-arm study: derive the two fixed cost thresholds from the validated pi_0 calibration of the SEPARATE cost model.

Reads artifacts/grpo/calibration_separate.json (+ .scores.jsonl): 256 calibration prompts x 4 pi_0 responses scored with the frozen
separate cost model (RM seed 42, best-by-val) -- the same artifact the seed-42 GRPO separate arm used (d = mu_C). Writes
  artifacts/sixarm/calibration_separate_d0.json   d0 = mu_C
  artifacts/sixarm/calibration_separate_d1.json   d1 = mu_C - 0.5 sigma_C
in the GRPO calibration format (arm, d, scorer_checkpoints, cost stats), plus prompt-cluster bootstrap SEs of mu/sigma/d and the
feasibility of d1 under pi_0. Thresholds are in RAW cost units and are never modified afterwards. CPU only; no models loaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from saferl.utils import read_json, read_jsonl, sha256_file, write_json  # noqa: E402

RULES = {"d0": 0.0, "d1": 0.5}


def derive(cal_path: Path, out_dir: Path, n_boot: int = 2000, seed: int = 0) -> dict:
    cal = read_json(cal_path)
    if cal["arm"] != "separate":
        raise ValueError("the six-arm thresholds use the SEPARATE cost model calibration")
    scores = read_jsonl(cal_path.with_suffix(".scores.jsonl"))
    c = np.array([s["cost"] for s in scores], dtype=np.float64)
    ph = np.array([s["prompt_hash"] for s in scores])
    mu, sigma = float(c.mean()), float(c.std())
    if abs(mu - float(cal["d"])) > 1e-6 or abs(sigma - float(cal["cost"]["std"])) > 1e-6:
        raise ValueError("scores file does not reproduce the calibration's mean/std")
    uniq = np.unique(ph); groups = [c[ph == u] for u in uniq]
    rng = np.random.default_rng(seed)
    boots = np.array([(lambda v: (v.mean(), v.std()))(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n_boot)])
    per_prompt_min = np.array([g.min() for g in groups]); per_prompt_mean = np.array([g.mean() for g in groups])
    out = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, alpha in RULES.items():
        d = mu - alpha * sigma
        d_boot = boots[:, 0] - alpha * boots[:, 1]
        rep = {**{k: v for k, v in cal.items() if k not in ("d", "note")}, "arm": "separate", "d": d, "d_rule": f"mu_C - {alpha} * sigma_C", "alpha": alpha,
               "units": "raw cost units of the frozen separate cost model", "mu_C": mu, "sigma_C": sigma,
               "uncertainty": {"mu_se": float(boots[:, 0].std()), "sigma_se": float(boots[:, 1].std()), "d_se": float(d_boot.std()),
                               "d_ci95": [float(np.percentile(d_boot, 2.5)), float(np.percentile(d_boot, 97.5))], "bootstrap": "prompt clusters (256), 2000 replicates"},
               "feasibility_under_pi0": {"frac_responses_at_or_below_d": float(np.mean(c <= d)), "frac_prompts_best_of_4_at_or_below_d": float(np.mean(per_prompt_min <= d)),
                                         "frac_prompts_mean_at_or_below_d": float(np.mean(per_prompt_mean <= d))},
               "source_calibration": str(cal_path), "source_sha256": sha256_file(cal_path),
               "note": "fixed for the whole study; C = 0 has no meaning (pairwise-trained cost model); matching d across arms matches modeled cost, not Beaver harm"}
        write_json(out_dir / f"calibration_separate_{name}.json", rep)
        out[name] = {"d": d, "d_se": rep["uncertainty"]["d_se"], **rep["feasibility_under_pi0"]}
    print({"mu_C": mu, "sigma_C": sigma, **out})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="artifacts/grpo/calibration_separate.json")
    ap.add_argument("--out", default="artifacts/sixarm")
    a = ap.parse_args()
    derive(Path(a.calibration), Path(a.out))
