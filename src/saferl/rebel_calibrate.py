"""Fixed reward/cost calibration for the REBEL stage.

1. ``generate``: ONE common calibration set from the initial policy pi_0: 256 calibration prompts x 4 completions
   (full-distribution sampling, fixed seeds) -> artifacts/rebel/calibration_generations.jsonl. Shared by every arm.
2. ``score`` (per arm): score those SAME responses with the arm's frozen scorer(s); for every head S record
   mu_S = mean, sigma_S = population std (ddof 0), bootstrap standard errors (resampling prompts), quantiles.
   Constrained arms: d_std (config, -0.5) and the equivalent raw threshold d_raw = mu_C + d_std sigma_C, plus the
   reference thresholds for alpha = 0 and 1 (d_std = 0, -1), and attainability statistics of d under pi_0.
   A degenerate estimate (non-finite or sigma < calibration_min_sigma) raises; nothing is replaced silently.
These statistics are frozen for the whole run (the trainer checks scorer fingerprints and the generations hash).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .grpo_rollout import generate_completions
from .grpo_scoring import FrozenScorers
from .modeling import load_model, load_tokenizer
from .rebel_core import raw_threshold
from .rebel_data import CALIBRATION_GEN_SEEDS, MANIFEST_DIR, load_rebel_prompts
from .utils import get_logger, load_config, pick_device, read_jsonl, sha256_file, write_json, write_jsonl

log = get_logger("saferl.rebel_calibrate")
ARM_HEADS = {"scalar": ["scalar"], "shared": ["reward", "cost"], "separate": ["reward", "cost"]}


def generate_calibration(cfg: Dict[str, Any], out_path: Optional[Path] = None, seeds: Optional[List[int]] = None, gen_batch: int = 32) -> Dict[str, Any]:
    device = pick_device(cfg.get("device", "auto"))
    seeds = seeds or CALIBRATION_GEN_SEEDS
    prompts = load_rebel_prompts(cfg["artifacts_dir"], "calibration")
    tok = load_tokenizer(cfg["policy_model"])
    model = load_model(cfg["policy_model"], device, torch.float32, attn_implementation=cfg.get("attn_implementation"))
    model.eval()
    gen = cfg["generation"]
    out_path = out_path or Path(cfg["artifacts_dir"]) / MANIFEST_DIR / "calibration_generations.jsonl"
    records: List[Dict[str, Any]] = []
    t0 = time.time()
    for k, seed in enumerate(seeds):
        for start in range(0, len(prompts), gen_batch):
            chunk = prompts[start:start + gen_batch]
            batch_seed = seed * 100003 + start
            recs = generate_completions(model, tok, [p["prompt"] for p in chunk], 1, gen["max_prompt_length"], gen["max_completion_length"], device,
                                        cfg.get("generation_precision") or cfg["precision"], gen["temperature"], gen["top_p"], gen["repetition_penalty"], seed=batch_seed)
            for p, r in zip(chunk, recs):
                records.append({"prompt_hash": p["prompt_hash"], "prompt": p["prompt"], "k": k, "seed": batch_seed, "response": r["text"],
                                "completion_ids": r["completion_ids"], "truncated": r["truncated"], "n_completion_tokens": r["n_completion_tokens"]})
            log.info("calibration generations: k=%d %d/%d prompts (%.0fs)", k, min(start + gen_batch, len(prompts)), len(prompts), time.time() - t0)
    write_jsonl(out_path, records)
    lens = np.array([r["n_completion_tokens"] for r in records])
    summary = {"n_prompts": len(prompts), "responses_per_prompt": len(seeds), "n_trajectories": len(records), "generation_seeds": seeds,
               "policy_model": cfg["policy_model"], "generation": gen, "device": device,
               "completion_tokens": {"mean": float(lens.mean()), "p50": float(np.median(lens)), "p95": float(np.percentile(lens, 95)), "max": int(lens.max())},
               "truncation_rate": float(np.mean([r["truncated"] for r in records])), "file": str(out_path), "file_sha256": sha256_file(out_path),
               "seconds": time.time() - t0}
    write_json(out_path.with_suffix(".summary.json"), summary)
    return summary


def head_stats(values: np.ndarray, prompt_ids: np.ndarray, n_boot: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """mu, sigma (population) and their bootstrap SEs resampling PROMPTS (clusters of 4 responses)."""
    mu, sigma = float(values.mean()), float(values.std())
    uniq = np.unique(prompt_ids)
    groups = [values[prompt_ids == u] for u in uniq]
    rng = np.random.default_rng(seed)
    mus, sigmas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        v = np.concatenate([groups[i] for i in idx])
        mus.append(v.mean()); sigmas.append(v.std())
    return {"mu": mu, "sigma": sigma, "mu_se": float(np.std(mus)), "sigma_se": float(np.std(sigmas)), "n": int(len(values)), "n_prompts": int(len(uniq)),
            "p5": float(np.percentile(values, 5)), "p25": float(np.percentile(values, 25)), "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)), "p95": float(np.percentile(values, 95)), "min": float(values.min()), "max": float(values.max()),
            "finite": bool(np.isfinite(values).all())}


def calibrate_arm(cfg: Dict[str, Any], generations_path: Optional[Path] = None, out_path: Optional[Path] = None) -> Dict[str, Any]:
    arm = cfg["arm"]
    if arm not in ARM_HEADS:
        raise ValueError(f"unknown arm {arm!r}")
    art = Path(cfg["artifacts_dir"]) / MANIFEST_DIR
    generations_path = generations_path or art / "calibration_generations.jsonl"
    recs = read_jsonl(generations_path)
    device = pick_device(cfg.get("device", "auto"))
    tok = load_tokenizer(cfg["policy_model"])
    gen = cfg["generation"]
    scorers = FrozenScorers(arm, cfg["scorer_checkpoints"], device, cfg["precision"], int(cfg.get("scorer_batch_size", 16)),
                            gen["max_prompt_length"], gen["max_prompt_length"] + gen["max_completion_length"])
    sc = scorers.score(tok, [r["prompt"] for r in recs], [r["response"] for r in recs])
    prompt_ids = np.array([r["prompt_hash"] for r in recs])
    min_sigma = float(cfg.get("calibration_min_sigma", 1e-3))
    heads: Dict[str, Any] = {}
    for h in ARM_HEADS[arm]:
        v = sc[h].numpy().astype(np.float64)
        st = head_stats(v, prompt_ids)
        if not st["finite"] or not np.isfinite(st["sigma"]) or st["sigma"] < min_sigma:
            raise ValueError(f"degenerate calibration for head {h!r}: sigma={st['sigma']} (min {min_sigma}), finite={st['finite']}")
        heads[h] = st
    report: Dict[str, Any] = {
        "arm": arm, "heads": heads, "standardization": "S~ = (S - mu) / sigma, sigma = population std over all calibration responses",
        "n_trajectories": len(recs), "n_prompts": int(len(np.unique(prompt_ids))), "generations_file": str(generations_path),
        "generations_sha256": sha256_file(generations_path), "scorer_checkpoints": scorers.fingerprints, "scorer_batch_size": int(cfg.get("scorer_batch_size", 16)),
        "precision": cfg["precision"], "device": device,
    }
    if arm in ("shared", "separate"):
        d_std = float(cfg["d_std"])
        mu_c, sig_c = heads["cost"]["mu"], heads["cost"]["sigma"]
        c_std = (sc["cost"].numpy() - mu_c) / sig_c
        per_prompt_min = np.array([c_std[prompt_ids == u].min() for u in np.unique(prompt_ids)])
        per_prompt_mean = np.array([c_std[prompt_ids == u].mean() for u in np.unique(prompt_ids)])
        report.update({
            "d_std": d_std, "d_raw": raw_threshold(mu_c, sig_c, d_std),
            "reference_thresholds": {f"alpha={a}": {"d_std": -a, "d_raw": raw_threshold(mu_c, sig_c, -a)} for a in (0.0, 0.5, 1.0)},
            "attainability_under_pi0": {
                "frac_responses_at_or_below_d": float(np.mean(c_std <= d_std)),
                "frac_prompts_best_of_4_at_or_below_d": float(np.mean(per_prompt_min <= d_std)),
                "mean_per_prompt_min_cost_std": float(per_prompt_min.mean()), "mean_per_prompt_mean_cost_std": float(per_prompt_mean.mean()),
                "note": "a mean below d is attainable if, e.g., choosing the best of 4 pi_0 samples would already satisfy it on most prompts; "
                        "matching the standardized threshold across arms does NOT imply matching true (Beaver) safety"},
            "reward_cost_spearman_pi0": float(_spearman(sc["reward"].numpy(), sc["cost"].numpy())),
        })
    out_path = out_path or art / f"calibration_{arm}.json"
    write_json(out_path, report)
    write_jsonl(out_path.with_suffix(".scores.jsonl"),
                ({"prompt_hash": r["prompt_hash"], "k": r["k"], "n_completion_tokens": r["n_completion_tokens"], "truncated": r["truncated"],
                  **{h: float(sc[h][i]) for h in ARM_HEADS[arm]}} for i, r in enumerate(recs)))
    log.info("calibrated %s: %s", arm, {h: {k: round(v, 4) for k, v in heads[h].items() if k in ("mu", "sigma", "mu_se", "sigma_se")} for h in heads})
    return report


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["generate", "score", "both"], default="both")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    gen_path = Path(cfg["artifacts_dir"]) / MANIFEST_DIR / "calibration_generations.jsonl"
    if args.stage in ("generate", "both"):
        if gen_path.exists() and args.stage == "both":
            log.info("calibration generations already exist at %s; reusing", gen_path)
        else:
            generate_calibration(cfg, gen_path)
    if args.stage in ("score", "both"):
        calibrate_arm(cfg, gen_path)


if __name__ == "__main__":
    main()
