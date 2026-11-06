"""Prompt manifests for the GRPO stage, derived from the FIXED DPO/RM split manifests.

* training prompts: the 10,000 training pairs deduplicated by normalized prompt -> 9,054 unique
  prompts (text of the first occurrence in pair_id order); same manifest for all three RL arms;
* calibration prompts (256) and monitoring prompts (256): two disjoint deterministic subsets of the
  unique validation prompts, chosen with data_seed = 42.

Prompt ORDER within a policy-training run is a seed-dependent permutation of the training manifest
(``prompt_order``); rollout batches of B prompts are taken in that order, the last batch may be partial.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .data import normalize_prompt
from .train_dpo import epoch_order
from .utils import read_jsonl, sha256_file, sha256_text, write_json, write_jsonl

MANIFEST_DIR = "grpo"
CALIBRATION_GEN_SEEDS = [1000, 1001, 1002, 1003]   # one fixed seed per response index k (pi_0 calibration generations)
MONITOR_GEN_SEEDS = [2000, 2001, 2002, 2003]       # one fixed seed per response index k (monitoring generations, all arms)
TRAIN_SAMPLE_GEN_SEEDS = [3000, 3001, 3002, 3003]  # pi_0 generations on the training-prompt sample (d representativeness check)


def unique_prompts(pairs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by normalized prompt; keep the first occurrence's text in sorted pair_id order."""
    seen: Dict[str, Dict[str, Any]] = OrderedDict()
    for p in sorted(pairs, key=lambda x: x["pair_id"]):
        key = normalize_prompt(p["prompt"])
        h = sha256_text(key)[:16]
        if h not in seen:
            seen[h] = {"prompt_hash": h, "prompt": p["prompt"], "first_pair_id": p["pair_id"], "n_pairs": 0,
                       "prompt_source": p.get("prompt_source")}
        seen[h]["n_pairs"] += 1
    return sorted(seen.values(), key=lambda x: x["prompt_hash"])


def build_manifests(artifacts_dir: str, data_seed: int = 42, n_calibration: int = 256, n_monitoring: int = 256,
                    n_train_sample: int = 256) -> Dict[str, Any]:
    art = Path(artifacts_dir)
    out = art / MANIFEST_DIR
    train_pairs = read_jsonl(art / "splits" / "train.pairs.jsonl")
    val_pairs = read_jsonl(art / "splits" / "val.pairs.jsonl")
    train_prompts = unique_prompts(train_pairs)
    val_prompts = unique_prompts(val_pairs)
    if len(val_prompts) < n_calibration + n_monitoring:
        raise ValueError(f"validation manifest has only {len(val_prompts)} unique prompts")
    rng = np.random.default_rng(data_seed)
    perm = rng.permutation(len(val_prompts))
    calib = [val_prompts[i] for i in sorted(perm[:n_calibration])]
    monitor = [val_prompts[i] for i in sorted(perm[n_calibration:n_calibration + n_monitoring])]
    assert not ({p["prompt_hash"] for p in calib} & {p["prompt_hash"] for p in monitor})
    assert not ({p["prompt_hash"] for p in train_prompts} & {p["prompt_hash"] for p in calib + monitor})
    # deterministic sample of TRAINING prompts (diagnostic only: is d, calibrated on validation prompts, representative?)
    rng_ts = np.random.default_rng(data_seed + 1)
    ts_idx = sorted(rng_ts.permutation(len(train_prompts))[: min(n_train_sample, len(train_prompts))])
    train_sample = [train_prompts[i] for i in ts_idx]
    files = {"train_prompts": out / "train_prompts.jsonl", "calibration_prompts": out / "calibration_prompts.jsonl",
             "monitoring_prompts": out / "monitoring_prompts.jsonl", "train_sample_prompts": out / "train_sample_prompts.jsonl"}
    write_jsonl(files["train_prompts"], train_prompts)
    write_jsonl(files["calibration_prompts"], calib)
    write_jsonl(files["monitoring_prompts"], monitor)
    write_jsonl(files["train_sample_prompts"], train_sample)
    info = {
        "data_seed": data_seed, "n_train_pairs": len(train_pairs), "n_train_prompts": len(train_prompts),
        "n_val_unique_prompts": len(val_prompts), "n_calibration_prompts": len(calib), "n_monitoring_prompts": len(monitor),
        "n_train_sample_prompts": len(train_sample),
        "calibration_generation_seeds": CALIBRATION_GEN_SEEDS, "monitoring_generation_seeds": MONITOR_GEN_SEEDS,
        "train_sample_generation_seeds": TRAIN_SAMPLE_GEN_SEEDS,
        "hashes": {k: sha256_file(v) for k, v in files.items()},
        "source_hashes": {"train.pairs.jsonl": sha256_file(art / "splits" / "train.pairs.jsonl"),
                          "val.pairs.jsonl": sha256_file(art / "splits" / "val.pairs.jsonl")},
    }
    write_json(out / "manifest_info.json", info)
    return info


def load_prompts(artifacts_dir: str, name: str) -> List[Dict[str, Any]]:
    return read_jsonl(Path(artifacts_dir) / MANIFEST_DIR / f"{name}_prompts.jsonl")


def prompt_order(n_prompts: int, seed: int) -> List[int]:
    """Seed-dependent permutation of the training manifest (one prompt epoch)."""
    return epoch_order(n_prompts, seed, 0)


def rollout_batches(order: Sequence[int], prompt_batch_size: int) -> List[List[int]]:
    """Consecutive batches of B prompt indices; the final batch may be smaller (never dropped)."""
    return [list(order[i:i + prompt_batch_size]) for i in range(0, len(order), prompt_batch_size)]
