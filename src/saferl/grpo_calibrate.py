"""Constraint-threshold calibration for the constrained GRPO arms.

Stage ``generate``: with the frozen INITIAL policy pi_0 (the original Qwen checkpoint), sample 4
responses for each of the 256 calibration prompts using the fixed generation seeds (one seed per
response index k, prompts processed in manifest order in fixed batches). Saved once; shared by both
constrained arms.

Stage ``score``: score those SAME 1,024 responses with one arm's frozen cost (and reward) model and
set   d_arm = mean cost over the 1,024 calibration trajectories.
Because the pairwise-trained cost model is identified only up to a constant b, d shifts by the same
b, leaving E[C] - d invariant. d is computed ONCE per arm and reused by every policy seed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .grpo_data import CALIBRATION_GEN_SEEDS, TRAIN_SAMPLE_GEN_SEEDS, load_prompts
from .grpo_rollout import generate_completions
from .grpo_scoring import FrozenScorers
from .modeling import load_model, load_tokenizer
from .utils import get_logger, load_config, pick_device, read_json, read_jsonl, sha256_file, write_json, write_jsonl

log = get_logger("saferl.grpo_calibrate")


def generate_calibration_set(cfg: Dict[str, Any], prompt_set: str = "calibration", out_path: Path = None,
                             seeds: List[int] = None, gen_batch: int = 32) -> Dict[str, Any]:
    """pi_0 generations for a prompt manifest: for each response index k, seed = seeds[k], prompts in
    manifest order, fixed batches of ``gen_batch`` prompts. Returns summary; writes one record per trajectory."""
    device = pick_device(cfg.get("device", "auto"))
    seeds = seeds or CALIBRATION_GEN_SEEDS
    prompts = load_prompts(cfg["artifacts_dir"], prompt_set)
    tok = load_tokenizer(cfg["policy_model"])
    model = load_model(cfg["policy_model"], device, torch.float32, attn_implementation=cfg.get("attn_implementation"))
    model.eval()
    gen = cfg["generation"]
    out_path = out_path or Path(cfg["artifacts_dir"]) / "grpo" / f"{prompt_set}_generations.jsonl"
    records: List[Dict[str, Any]] = []
    t0 = time.time()
    for k, seed in enumerate(seeds):
        for start in range(0, len(prompts), gen_batch):
            chunk = prompts[start:start + gen_batch]
            batch_seed = seed * 100003 + start  # fixed per (k, batch position)
            recs = generate_completions(model, tok, [p["prompt"] for p in chunk], 1, gen["max_prompt_length"],
                                        gen["max_completion_length"], device, cfg.get("generation_precision") or cfg["precision"], gen["temperature"], gen["top_p"],
                                        gen["repetition_penalty"], seed=batch_seed)
            for p, r in zip(chunk, recs):
                records.append({"prompt_hash": p["prompt_hash"], "prompt": p["prompt"], "k": k, "seed": batch_seed,
                                "response": r["text"], "completion_ids": r["completion_ids"], "truncated": r["truncated"],
                                "n_completion_tokens": r["n_completion_tokens"]})
            log.info("%s generations: k=%d %d/%d prompts (%.0fs)", prompt_set, k, min(start + gen_batch, len(prompts)), len(prompts), time.time() - t0)
    write_jsonl(out_path, records)
    lens = np.array([r["n_completion_tokens"] for r in records])
    summary = {"prompt_set": prompt_set, "n_prompts": len(prompts), "responses_per_prompt": len(seeds), "n_trajectories": len(records),
               "generation_seeds": seeds, "policy_model": cfg["policy_model"], "generation": gen, "device": device,
               "completion_tokens": {"mean": float(lens.mean()), "p50": float(np.median(lens)), "p95": float(np.percentile(lens, 95)),
                                     "max": int(lens.max())},
               "truncation_rate": float(np.mean([r["truncated"] for r in records])), "file": str(out_path),
               "file_sha256": sha256_file(out_path), "seconds": time.time() - t0}
    write_json(out_path.with_suffix(".summary.json"), summary)
    log.info("wrote %s: %s", out_path, {k: v for k, v in summary.items() if k in ("n_trajectories", "truncation_rate", "completion_tokens")})
    return summary


def calibrate_d(cfg: Dict[str, Any], generations_path: Path = None, out_path: Path = None) -> Dict[str, Any]:
    """Score the saved pi_0 calibration trajectories with this arm's frozen scorers; d = mean cost."""
    arm = cfg["arm"]
    if arm not in ("shared", "separate"):
        raise ValueError("calibration is only defined for the constrained arms (shared, separate)")
    device = pick_device(cfg.get("device", "auto"))
    generations_path = generations_path or Path(cfg["artifacts_dir"]) / "grpo" / "calibration_generations.jsonl"
    recs = read_jsonl(generations_path)
    tok = load_tokenizer(cfg["policy_model"])
    scorers = FrozenScorers(arm, cfg["scorer_checkpoints"], device, cfg["precision"], int(cfg.get("scorer_batch_size", 16)),
                            cfg["generation"]["max_prompt_length"], cfg["generation"]["max_prompt_length"] + cfg["generation"]["max_completion_length"])
    sc = scorers.score(tok, [r["prompt"] for r in recs], [r["response"] for r in recs])
    c = sc["cost"].numpy()
    r = sc["reward"].numpy()
    d = float(c.mean())
    report = {
        "arm": arm, "d": d, "n_trajectories": len(recs), "generations_file": str(generations_path),
        "generations_sha256": sha256_file(generations_path), "scorer_checkpoints": scorers.fingerprints,
        "cost": {"mean": d, "std": float(c.std()), "p5": float(np.percentile(c, 5)), "median": float(np.median(c)),
                 "p95": float(np.percentile(c, 95)), "min": float(c.min()), "max": float(c.max())},
        "reward": {"mean": float(r.mean()), "std": float(r.std()), "p5": float(np.percentile(r, 5)), "median": float(np.median(r)),
                   "p95": float(np.percentile(r, 95)), "min": float(r.min()), "max": float(r.max())},
        "note": "cost model trained with pairwise BT only: C is identified up to an additive constant; d is the initial-policy mean "
                "cost under THIS cost model and is reused unchanged for every policy seed of this arm. C = 0 has no safety meaning.",
    }
    out_path = out_path or Path(cfg["artifacts_dir"]) / "grpo" / f"calibration_{arm}.json"
    write_json(out_path, report)
    write_jsonl(out_path.with_suffix(".scores.jsonl"), ({"prompt_hash": x["prompt_hash"], "k": x["k"], "cost": float(ci), "reward": float(ri)}
                                                         for x, ci, ri in zip(recs, c, r)))
    log.info("calibrated d_%s = %.6f (cost mean %.4f std %.4f p5 %.4f median %.4f p95 %.4f; reward mean %.4f)", arm, d, d,
             report["cost"]["std"], report["cost"]["p5"], report["cost"]["median"], report["cost"]["p95"], report["reward"]["mean"])
    return report


def representativeness_check(cfg: Dict[str, Any], calibration_path: Path = None, gen_path: Path = None, out_path: Path = None) -> Dict[str, Any]:
    """Is d (mean pi_0 cost on the 256 VALIDATION calibration prompts) representative of TRAINING prompts?
    Scores pi_0 generations on the deterministic training-prompt sample with the same frozen cost model and
    reports mean cost - d (also in units of std(C) and of the standard error). Report only; d is never changed."""
    arm = cfg["arm"]
    art = Path(cfg["artifacts_dir"]) / "grpo"
    calibration_path = calibration_path or art / f"calibration_{arm}.json"
    cal = read_json(calibration_path)
    gen_path = gen_path or art / "train_sample_generations.jsonl"
    if not gen_path.exists():
        generate_calibration_set(cfg, "train_sample", gen_path, seeds=TRAIN_SAMPLE_GEN_SEEDS)
    recs = read_jsonl(gen_path)
    device = pick_device(cfg.get("device", "auto"))
    tok = load_tokenizer(cfg["policy_model"])
    scorers = FrozenScorers(arm, cfg["scorer_checkpoints"], device, cfg["precision"], int(cfg.get("scorer_batch_size", 16)),
                            cfg["generation"]["max_prompt_length"], cfg["generation"]["max_prompt_length"] + cfg["generation"]["max_completion_length"])
    for k, fp in cal["scorer_checkpoints"].items():  # same frozen cost model as the calibration
        mine = scorers.fingerprints.get(k, {})
        if fp.get("model.safetensors") != mine.get("model.safetensors"):
            raise ValueError(f"representativeness check must use the calibration's {k} checkpoint")
    sc = scorers.score(tok, [r["prompt"] for r in recs], [r["response"] for r in recs])
    c = sc["cost"].numpy(); r = sc["reward"].numpy()
    d = float(cal["d"]); cal_std = float(cal["cost"]["std"])
    diff = float(c.mean()) - d
    se = float(np.sqrt(cal_std ** 2 / cal["n_trajectories"] + c.std() ** 2 / len(c)))
    report = {
        "arm": arm, "d": d, "n_calibration_trajectories": cal["n_trajectories"], "calibration_cost_std": cal_std,
        "train_sample": {"n_trajectories": len(recs), "n_prompts": len({x["prompt_hash"] for x in recs}), "generations_file": str(gen_path),
                         "generations_sha256": sha256_file(gen_path), "generation_seeds": TRAIN_SAMPLE_GEN_SEEDS,
                         "cost": {"mean": float(c.mean()), "std": float(c.std()), "p5": float(np.percentile(c, 5)), "median": float(np.median(c)),
                                  "p95": float(np.percentile(c, 95))},
                         "reward_mean": float(r.mean())},
        "train_sample_mean_cost_minus_d": diff, "diff_in_calibration_std_units": diff / cal_std if cal_std > 0 else None,
        "diff_in_standard_errors": diff / se if se > 0 else None,
        "note": "report only: d stays the validation-prompt calibration; a material difference must be flagged, not corrected silently",
    }
    out_path = out_path or art / f"calibration_{arm}_representativeness.json"
    write_json(out_path, report)
    log.info("representativeness %s: train-sample mean cost %.4f vs d %.4f -> diff %.4f (%.2f calibration std, %.2f SE)", arm,
             float(c.mean()), d, diff, report["diff_in_calibration_std_units"] or 0, report["diff_in_standard_errors"] or 0)
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["generate", "score", "both", "representativeness"], default="both")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    gen_path = Path(cfg["artifacts_dir"]) / "grpo" / "calibration_generations.jsonl"
    if args.stage in ("generate", "both"):
        if gen_path.exists() and args.stage == "both":
            log.info("calibration generations already exist at %s; reusing", gen_path)
        else:
            generate_calibration_set(cfg, "calibration", gen_path)
    if args.stage in ("score", "both"):
        calibrate_d(cfg, gen_path)
    if args.stage == "representativeness":
        representativeness_check(cfg)


if __name__ == "__main__":
    main()
