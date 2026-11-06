"""Common evaluation pipeline: manifest, deterministic per-prompt seeds, fixed batches and shards.

* Evaluation prompts: the ``test_representative`` split deduplicated by normalized prompt (1,960 unique
  prompts, each weighing 1), saved once to ``artifacts/eval/eval_prompts.jsonl``; every method and seed
  uses exactly this manifest.
* Per-prompt generation seed = sha256(f"{eval_seed}:{prompt_hash}"): identical for every method and
  policy seed, independent of batch composition and sharding.
* Generation batches are FIXED (``batch_size`` consecutive prompts in manifest order); a shard is a
  contiguous range of whole batches, so batch composition, and therefore every sampled token, is
  identical whether the run uses 1 shard or N shards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .grpo_data import unique_prompts
from .utils import read_jsonl, sha256_file, sha256_text, write_json, write_jsonl

EVAL_DIR = "eval"
EVALUATOR_TEMPLATE = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}"


def build_eval_manifest(artifacts_dir: str, source_split: str = "test_representative") -> Dict[str, Any]:
    art = Path(artifacts_dir)
    pairs = read_jsonl(art / "splits" / f"{source_split}.pairs.jsonl")
    prompts = unique_prompts(pairs)  # sorted by prompt_hash, deterministic
    out = art / EVAL_DIR / "eval_prompts.jsonl"
    write_jsonl(out, prompts)
    info = {"source_split": source_split, "n_pairs": len(pairs), "n_prompts": len(prompts), "file": str(out),
            "sha256": sha256_file(out), "source_sha256": sha256_file(art / "splits" / f"{source_split}.pairs.jsonl")}
    write_json(art / EVAL_DIR / "manifest_info.json", info)
    return info


def load_eval_prompts(artifacts_dir: str) -> List[Dict[str, Any]]:
    return read_jsonl(Path(artifacts_dir) / EVAL_DIR / "eval_prompts.jsonl")


def prompt_seed(eval_seed: int, prompt_hash: str) -> int:
    """Deterministic 63-bit generation seed for one prompt; same for every method and policy seed."""
    return int(sha256_text(f"{int(eval_seed)}:{prompt_hash}")[:16], 16) % (2 ** 63 - 1)


def fixed_batches(n_prompts: int, batch_size: int) -> List[List[int]]:
    return [list(range(i, min(i + batch_size, n_prompts))) for i in range(0, n_prompts, batch_size)]


def shard_batches(n_prompts: int, batch_size: int, num_shards: int, shard_id: int) -> List[List[int]]:
    """Shard = contiguous range of whole fixed batches (np.array_split over batch indices)."""
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id {shard_id} out of range for {num_shards} shards")
    batches = fixed_batches(n_prompts, batch_size)
    if num_shards > len(batches):
        raise ValueError(f"{num_shards} shards but only {len(batches)} batches of {batch_size} prompts")
    parts = np.array_split(np.arange(len(batches)), num_shards)
    return [batches[i] for i in parts[shard_id]]


def shard_prompt_indices(n_prompts: int, batch_size: int, num_shards: int, shard_id: int) -> List[int]:
    return [i for b in shard_batches(n_prompts, batch_size, num_shards, shard_id) for i in b]


def method_checkpoint(method_cfg: Dict[str, Any], seed: int) -> str:
    return str(method_cfg["checkpoint"]).format(seed=seed)


CHECKPOINT_RULES = ("pretrained", "dpo_final_2_epochs", "grpo_final_1_prompt_epoch", "rebel_final_1_prompt_epoch")


def verify_checkpoint_rule(path: str, rule: str) -> Dict[str, Any]:
    """Enforce the pre-specified PRIMARY checkpoint rule from the checkpoint's own ``trainer_state.json``:
      dpo_final_2_epochs        -> end-of-training DPO checkpoint, epoch == max_epochs == 2 exactly;
      grpo_final_1_prompt_epoch -> final GRPO checkpoint after exactly one prompt epoch (all prompts processed);
      pretrained                -> Hub id / untrained checkpoint (no trainer state).
    Any path whose last component is ``best`` (validation-selected) is refused for every rule."""
    if rule not in CHECKPOINT_RULES and not rule.startswith("after_batches:"):
        raise ValueError(f"unknown checkpoint rule {rule!r}")
    p = Path(path)
    if p.name == "best" or "/best/" in str(path).replace("\\", "/") + "/":
        raise ValueError(f"primary evaluation refuses the validation-selected checkpoint {path}")
    if rule == "pretrained":
        if p.is_dir() and (p / "trainer_state.json").exists():
            raise ValueError(f"{path} is a trained checkpoint but the rule is 'pretrained'")
        return {"rule": rule}
    state_file = p / "trainer_state.json"
    if not state_file.exists():
        raise ValueError(f"{path}: trainer_state.json missing; cannot verify rule {rule}")
    import json

    st = json.loads(state_file.read_text())
    if rule.startswith("after_batches:"):
        # Fixed-budget rule (six-arm study): the checkpoint written after exactly N rollout batches of the full
        # (resumable) schedule; prompts processed must equal the batches' prompt count recorded by the trainer.
        n = int(rule.split(":", 1)[1])
        if int(st.get("next_batch_index", -1)) != n or int(st.get("batch_index", -1)) != n:
            raise ValueError(f"{path}: not the checkpoint after {n} batches (state={ {k: st.get(k) for k in ('batch_index', 'next_batch_index', 'n_prompts')} })")
        return {"rule": rule, "trainer_state": {k: st.get(k) for k in ("batch_index", "next_batch_index", "opt_step", "n_prompts", "algorithm", "arm", "seed")}}
    if rule == "dpo_final_2_epochs":
        if not st.get("is_epoch_end") or abs(float(st.get("epoch", -1)) - 2.0) > 1e-6:
            raise ValueError(f"{path}: not the end-of-epoch-2 DPO checkpoint (state={ {k: st.get(k) for k in ('step', 'epoch', 'is_epoch_end')} })")
    elif rule in ("grpo_final_1_prompt_epoch", "rebel_final_1_prompt_epoch"):
        if int(st.get("prompt_epochs", 0)) != 1 or "n_prompts" not in st or st.get("intermediate"):
            raise ValueError(f"{path}: not a final one-prompt-epoch checkpoint (state keys={sorted(st)[:8]})")
        if rule == "rebel_final_1_prompt_epoch" and st.get("algorithm") != "rebel":
            raise ValueError(f"{path}: rule {rule} requires a REBEL checkpoint (algorithm={st.get('algorithm')})")
        cfg_file = p.parent.parent / "config_resolved.json"
        if cfg_file.exists():
            n_train = json.loads(cfg_file.read_text()).get("n_train_prompts")
            if n_train is not None and int(st["n_prompts"]) != int(n_train):
                raise ValueError(f"{path}: processed {st['n_prompts']} prompts, expected the full epoch of {n_train}")
    return {"rule": rule, "trainer_state": {k: st.get(k) for k in ("step", "epoch", "is_epoch_end", "opt_step", "prompt_epochs", "n_prompts", "arm", "seed")}}


def run_dir(cfg: Dict[str, Any], method: str, seed: int) -> Path:
    return Path(cfg["out_dir"]) / method / f"seed{seed}"


def shard_file(d: Path, kind: str, num_shards: int, shard_id: int) -> Path:
    return d / f"{kind}.shard{shard_id:03d}of{num_shards:03d}.jsonl"
