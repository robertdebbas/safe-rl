"""Evaluation-response generation with a deterministic seed PER PROMPT.

Hugging Face ``generate`` draws one random stream for the whole batch, so a prompt's sample would
depend on its batch mates. Here each prompt owns a CPU ``torch.Generator`` seeded from
(eval_seed, prompt_hash); at every step we draw u_i ~ U(0,1) from that generator and pick the token by
inverse CDF on the (temperature / top-p adjusted) next-token distribution. The random draws are therefore
identical for every method, every policy seed and every shard layout; the only cross-prompt coupling left
is bf16 arithmetic on padded batches, which is why generation batches are fixed (see eval_common).

Outputs one record per prompt: prompt_hash, prompt, response, completion_ids, n_completion_tokens,
truncated, seed, method, policy_seed, checkpoint.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .eval_common import (load_eval_prompts, method_checkpoint, prompt_seed, run_dir, shard_batches, shard_file,
                          verify_checkpoint_rule)
from .grpo_rollout import encode_prompts_left_padded, eos_ids
from .modeling import autocast_ctx, load_model, load_tokenizer
from .utils import get_logger, load_config, pick_device, write_json, write_jsonl

log = get_logger("saferl.eval_generate")


def nucleus_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative probability >= top_p (always keeps the top token)."""
    if top_p >= 1.0:
        return probs
    sp, idx = torch.sort(probs, dim=-1, descending=True)
    cum = sp.cumsum(-1)
    remove = cum - sp > top_p
    sp = torch.where(remove, torch.zeros_like(sp), sp)
    out = torch.zeros_like(probs).scatter(-1, idx, sp)
    return out / out.sum(-1, keepdim=True)


@torch.no_grad()
def sample_with_prompt_seeds(model, tokenizer, prompts: Sequence[str], seeds: Sequence[int], max_prompt_length: int,
                             max_new_tokens: int, device: str, precision: str = "bf16", temperature: float = 1.0,
                             top_p: float = 1.0) -> List[Dict[str, Any]]:
    model.eval()
    enc = encode_prompts_left_padded(tokenizer, prompts, max_prompt_length, device)
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    B = len(prompts)
    gens = [torch.Generator().manual_seed(int(s)) for s in seeds]      # CPU generators: device independent
    stop = set(eos_ids(tokenizer))
    pad = tokenizer.pad_token_id
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    cur_ids, cur_attn, cur_pos, past = input_ids, attn, pos, None
    finished = [False] * B
    out: List[List[int]] = [[] for _ in range(B)]
    for _ in range(max_new_tokens):
        with autocast_ctx(device, precision):
            o = model(input_ids=cur_ids, attention_mask=cur_attn, position_ids=cur_pos, past_key_values=past, use_cache=True)
        logits = o.logits[:, -1, :].float()
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        probs = nucleus_filter(probs, top_p)
        u = torch.tensor([float(torch.rand(1, generator=g)[0]) for g in gens], dtype=torch.float32, device=probs.device)
        cdf = probs.cumsum(-1)
        nxt = torch.searchsorted(cdf, u.unsqueeze(1)).squeeze(1).clamp(max=probs.shape[-1] - 1)
        nxt_list = nxt.tolist()
        for i in range(B):
            if finished[i]:
                nxt_list[i] = pad
            else:
                out[i].append(nxt_list[i])
                if nxt_list[i] in stop:
                    finished[i] = True
        past = o.past_key_values
        cur_ids = torch.tensor(nxt_list, device=device).unsqueeze(1)
        cur_attn = torch.cat([cur_attn, torch.ones((B, 1), dtype=cur_attn.dtype, device=device)], dim=1)
        cur_pos = cur_pos[:, -1:] + 1
        if all(finished):
            break
    recs = []
    for i in range(B):
        comp = out[i]
        truncated = not (comp and comp[-1] in stop)
        recs.append({"prompt": prompts[i], "completion_ids": comp, "response": tokenizer.decode([t for t in comp if t not in stop], skip_special_tokens=True),
                     "n_completion_tokens": len(comp), "truncated": truncated, "seed": int(seeds[i])})
    return recs


def checkpoint_fingerprint(ckpt: str) -> Dict[str, Any]:
    """Local directory: sha256 of model.safetensors; Hub id: resolved commit sha (from the local snapshot if cached)."""
    from .utils import sha256_file

    p = Path(ckpt)
    if p.is_dir():
        f = p / "model.safetensors"
        return {"type": "local", "path": str(p.resolve()), "model_safetensors_sha256": sha256_file(f) if f.exists() else None}
    try:
        from huggingface_hub import HfApi

        return {"type": "hub", "repo": ckpt, "commit_sha": HfApi().model_info(ckpt).sha}
    except Exception as e:  # offline: fall back to the snapshot directory name
        try:
            from huggingface_hub import snapshot_download

            snap = snapshot_download(ckpt, local_files_only=True)
            return {"type": "hub", "repo": ckpt, "commit_sha": Path(snap).name, "note": "from local cache"}
        except Exception:
            return {"type": "hub", "repo": ckpt, "commit_sha": None, "error": repr(e)}


def generate_shard(cfg: Dict[str, Any], method: str, seed: int, num_shards: int, shard_id: int, overwrite: bool = False) -> Dict[str, Any]:
    mcfg = cfg["methods"][method]
    ckpt = method_checkpoint(mcfg, seed)
    rule_check = verify_checkpoint_rule(ckpt, mcfg["checkpoint_rule"])  # PRIMARY rule: final checkpoints only, never 'best'
    gen = cfg["generation"]
    prompts = load_eval_prompts(cfg["artifacts_dir"])
    batches = shard_batches(len(prompts), int(gen["batch_size"]), num_shards, shard_id)
    d = run_dir(cfg, method, seed)
    out_path = shard_file(d, "responses", num_shards, shard_id)
    if out_path.exists() and not overwrite:
        log.info("%s exists; skipping (use --overwrite to regenerate)", out_path)
        return {"skipped": True, "file": str(out_path)}
    device = pick_device(cfg.get("device", "auto"))
    tok = load_tokenizer(ckpt)
    model = load_model(ckpt, device, torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32,
                       attn_implementation=cfg.get("attn_implementation"))
    t0 = time.time()
    records: List[Dict[str, Any]] = []
    n_done = 0
    for bi, b in enumerate(batches):
        ps = [prompts[i] for i in b]
        seeds = [prompt_seed(int(cfg["eval_seed"]), p["prompt_hash"]) for p in ps]
        recs = sample_with_prompt_seeds(model, tok, [p["prompt"] for p in ps], seeds, gen["max_prompt_length"], gen["max_new_tokens"],
                                        device, cfg["precision"], gen["temperature"], gen["top_p"])
        for p, r in zip(ps, recs):
            records.append({"prompt_hash": p["prompt_hash"], **r, "method": method, "policy_seed": seed, "checkpoint": ckpt})
        n_done += len(b)
        if (bi + 1) % 5 == 0 or bi + 1 == len(batches):
            log.info("%s seed %s shard %d/%d: %d/%d prompts (%.0fs)", method, seed, shard_id, num_shards, n_done,
                     sum(len(x) for x in batches), time.time() - t0)
    # manifest index for ordering at merge time
    index = {p["prompt_hash"]: i for i, p in enumerate(prompts)}
    for r in records:
        r["manifest_index"] = index[r["prompt_hash"]]
    write_jsonl(out_path, records)
    from .utils import read_json as _rj
    manifest_info = _rj(Path(cfg["artifacts_dir"]) / "eval" / "manifest_info.json")
    summary = {"method": method, "policy_seed": seed, "checkpoint": ckpt, "checkpoint_fingerprint": checkpoint_fingerprint(ckpt),
               "checkpoint_rule": rule_check,
               "manifest_sha256": manifest_info["sha256"], "n_manifest_prompts": manifest_info["n_prompts"],
               "num_shards": num_shards, "shard_id": shard_id,
               "n_prompts": len(records), "n_truncated": sum(r["truncated"] for r in records),
               "mean_completion_tokens": sum(r["n_completion_tokens"] for r in records) / max(1, len(records)),
               "seconds": time.time() - t0, "device": device, "generation": gen, "eval_seed": cfg["eval_seed"], "file": str(out_path)}
    write_json(out_path.with_suffix(".summary.json"), summary)
    log.info("wrote %s (%d prompts, truncation %.3f, %.0fs)", out_path, len(records), summary["n_truncated"] / max(1, len(records)), summary["seconds"])
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True, help="policy seed (0 for single-checkpoint methods such as base)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    generate_shard(cfg, args.method, args.seed, args.num_shards, args.shard_id, args.overwrite)


if __name__ == "__main__":
    main()
