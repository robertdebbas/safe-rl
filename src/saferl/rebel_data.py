"""Policy-training prompt manifests for the REBEL stage (and its matched GRPO comparison).

Source pool = prompts of the OFFICIAL PKU-SafeRLHF train split (pinned revision) that were NEVER used anywhere
else: not in the RM/DPO train or val splits, not in either held-out test split, and not any official test
prompt at all (even rows the validity filter dropped). Disjointness is enforced at the PROMPT level
(normalized-prompt sha256 key, the same key used by the rest of the pipeline).

From that pool, data_seed-deterministic stratified samples (by prompt_source, quotas matching the source mix
of the previous 9,054 GRPO training prompts) give three disjoint manifests:
    artifacts/rebel/train_prompts.jsonl        (target 9,054; same prompts and order for every arm)
    artifacts/rebel/calibration_prompts.jsonl  (256; pi_0 calibration of mu/sigma per scorer)
    artifacts/rebel/monitoring_prompts.jsonl   (256; monitoring / generation dumps)
``manifest_info.json`` records counts, per-source quotas, every pairwise hash intersection (all must be 0)
and the file hashes.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .data import filter_valid, largest_remainder, normalize_prompt
from .utils import read_jsonl, sha256_file, sha256_text, write_json, write_jsonl

MANIFEST_DIR = "rebel"
CALIBRATION_GEN_SEEDS = [5000, 5001, 5002, 5003]   # pi_0 calibration generations, one seed per response index k
MONITOR_GEN_SEEDS = [6000, 6001, 6002, 6003]       # monitoring generations (all arms)
DUMP_GEN_SEED = 7000                               # light generation dumps every N batches (32 prompts x 1)
SPLIT_FILES = ("train", "val", "test_representative", "test_balanced")


def prompt_key(prompt: str) -> str:
    return sha256_text(normalize_prompt(prompt))[:16]


def used_prompt_keys(artifacts_dir: str) -> Dict[str, Set[str]]:
    """Prompt keys of every existing split (RM/DPO train+val, both test splits) and of the old GRPO manifests."""
    art = Path(artifacts_dir)
    out: Dict[str, Set[str]] = {}
    for s in SPLIT_FILES:
        out[f"split:{s}"] = {prompt_key(p["prompt"]) for p in read_jsonl(art / "splits" / f"{s}.pairs.jsonl")}
    for name in ("train", "calibration", "monitoring"):
        f = art / "grpo" / f"{name}_prompts.jsonl"
        if f.exists():
            out[f"grpo:{name}"] = {p["prompt_hash"] for p in read_jsonl(f)}
    return out


def eligible_pool(train_pairs: Sequence[Dict[str, Any]], official_test_pairs: Sequence[Dict[str, Any]], forbidden: Set[str],
                  prompt_token_len=None, max_prompt_length: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Unique prompts of the valid official train pairs minus forbidden keys minus ALL official test prompt keys."""
    valid, validity = filter_valid(list(train_pairs))
    test_keys = {prompt_key(p["prompt"]) for p in official_test_pairs}
    seen: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for p in sorted(valid, key=lambda x: x["pair_id"]):
        k = prompt_key(p["prompt"])
        if k not in seen:
            seen[k] = {"prompt_hash": k, "prompt": p["prompt"], "first_pair_id": p["pair_id"], "n_pairs": 0, "prompt_source": p.get("prompt_source")}
        seen[k]["n_pairs"] += 1
    n_unique = len(seen)
    pool = [v for k, v in seen.items() if k not in forbidden and k not in test_keys]
    n_after_exclusion = len(pool)
    n_too_long = 0
    if prompt_token_len is not None and max_prompt_length:
        kept = []
        for v in pool:
            if prompt_token_len(v["prompt"]) <= max_prompt_length:
                kept.append(v)
            else:
                n_too_long += 1
        pool = kept
    pool.sort(key=lambda x: x["prompt_hash"])
    report = {"valid_train_pairs": len(valid), "validity": validity, "unique_train_prompts": n_unique,
              "removed_forbidden_or_test": n_unique - n_after_exclusion, "removed_prompt_too_long": n_too_long, "eligible_prompts": len(pool),
              "eligible_by_source": dict(Counter(v["prompt_source"] for v in pool))}
    return pool, report


def source_quotas(reference_sources: Dict[str, int], n: int) -> Dict[str, int]:
    """Integer quotas (largest remainder) reproducing the reference source proportions for n prompts."""
    tot = sum(reference_sources.values())
    return largest_remainder({s: c / tot for s, c in reference_sources.items()}, n)


def stratified_take(pool: Sequence[Dict[str, Any]], quotas: Dict[str, int], rng: np.random.Generator, taken: Set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src in sorted(quotas):
        q = int(quotas[src])
        cand = [p for p in pool if p["prompt_source"] == src and p["prompt_hash"] not in taken]
        if len(cand) < q:
            raise ValueError(f"source {src!r}: need {q} prompts, only {len(cand)} unused eligible")
        idx = sorted(rng.permutation(len(cand))[:q].tolist())
        for i in idx:
            out.append(cand[i]); taken.add(cand[i]["prompt_hash"])
    return sorted(out, key=lambda x: x["prompt_hash"])


def build_rebel_manifests(artifacts_dir: str, train_pairs: Sequence[Dict[str, Any]], official_test_pairs: Sequence[Dict[str, Any]],
                          data_seed: int = 42, n_train: int = 9054, n_calibration: int = 256, n_monitoring: int = 256,
                          reference_sources: Optional[Dict[str, int]] = None, prompt_token_len=None, max_prompt_length: Optional[int] = None,
                          dataset_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    art = Path(artifacts_dir)
    out = art / MANIFEST_DIR
    used = used_prompt_keys(artifacts_dir)
    forbidden: Set[str] = set().union(*used.values()) if used else set()
    pool, pool_report = eligible_pool(train_pairs, official_test_pairs, forbidden, prompt_token_len, max_prompt_length)
    if reference_sources is None:
        ref_file = art / "grpo" / "train_prompts.jsonl"
        reference_sources = dict(Counter(p["prompt_source"] for p in read_jsonl(ref_file))) if ref_file.exists() else dict(pool_report["eligible_by_source"])
    rng = np.random.default_rng(data_seed)
    taken: Set[str] = set()
    quotas = {"train": source_quotas(reference_sources, n_train), "calibration": source_quotas(reference_sources, n_calibration),
              "monitoring": source_quotas(reference_sources, n_monitoring)}
    sets = {name: stratified_take(pool, quotas[name], rng, taken) for name in ("train", "calibration", "monitoring")}
    keys = {name: {p["prompt_hash"] for p in s} for name, s in sets.items()}
    inter: Dict[str, int] = {}
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter[f"{a}&{b}"] = len(keys[a] & keys[b])
        for u, uk in used.items():
            inter[f"{a}&{u}"] = len(keys[a] & uk)
        inter[f"{a}&official_test"] = len(keys[a] & {prompt_key(p["prompt"]) for p in official_test_pairs})
    if any(v != 0 for v in inter.values()):
        raise RuntimeError(f"prompt leakage in REBEL manifests: {inter}")
    out.mkdir(parents=True, exist_ok=True)
    files = {name: out / f"{name}_prompts.jsonl" for name in sets}
    for name, s in sets.items():
        write_jsonl(files[name], s)
    info = {
        "data_seed": data_seed, "dataset": dataset_meta, "pool": pool_report, "reference_sources": reference_sources, "quotas": quotas,
        "counts": {name: len(s) for name, s in sets.items()}, "by_source": {name: dict(Counter(p["prompt_source"] for p in s)) for name, s in sets.items()},
        "hash_intersections": inter, "forbidden_prompt_keys": {k: len(v) for k, v in used.items()},
        "calibration_generation_seeds": CALIBRATION_GEN_SEEDS, "monitoring_generation_seeds": MONITOR_GEN_SEEDS, "dump_generation_seed": DUMP_GEN_SEED,
        "hashes": {name: sha256_file(f) for name, f in files.items()},
        "note": "prompt-level disjointness from every RM/DPO split, both test splits, all official test prompts and the old GRPO manifests; "
                "same prompts and order (prompt_order(seed)) for every arm",
    }
    write_json(out / "manifest_info.json", info)
    return info


def load_rebel_prompts(artifacts_dir: str, name: str) -> List[Dict[str, Any]]:
    return read_jsonl(Path(artifacts_dir) / MANIFEST_DIR / f"{name}_prompts.jsonl")


def build_from_official(artifacts_dir: str, data_config: str = "configs/data.yaml", **kw) -> Dict[str, Any]:
    """Entry point: official dataset at the pinned revision + tokenizer length check (prompt <= max_prompt_length)."""
    from .data import dataset_to_pairs, load_official
    from .formatting import tokenize_prompt
    from .modeling import load_tokenizer
    from .prepare import load_length_limits
    from .utils import load_config

    dcfg = load_config(data_config, [])
    ds, meta = load_official(dcfg["revision"], dcfg.get("config", "default"), dcfg.get("dataset", "PKU-Alignment/PKU-SafeRLHF"))
    train_pairs = dataset_to_pairs(ds["train"], "train")
    test_pairs = dataset_to_pairs(ds["test"], "test")
    tok = load_tokenizer(dcfg["tokenizer"])
    lim = load_length_limits(artifacts_dir)
    return build_rebel_manifests(artifacts_dir, train_pairs, test_pairs, data_seed=int(dcfg["data_seed"]),
                                 prompt_token_len=lambda p: len(tokenize_prompt(tok, p)), max_prompt_length=int(lim["max_prompt_length"]),
                                 dataset_meta={k: meta[k] for k in ("dataset", "config", "revision", "official_split_sizes")}, **kw)


if __name__ == "__main__":  # python -m saferl.rebel_data [artifacts_dir]
    import json
    import sys

    print(json.dumps(build_from_official(sys.argv[1] if len(sys.argv) > 1 else "artifacts"), indent=1))
