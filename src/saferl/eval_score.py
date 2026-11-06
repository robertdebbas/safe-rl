"""Scoring saved (prompt, response) pairs with the FROZEN external PKU evaluators.

    helpfulness: PKU-Alignment/beaver-7b-unified-reward   (higher = more helpful)
    harmfulness: PKU-Alignment/beaver-7b-unified-cost     (higher = more harmful)

Both are ``LlamaForScore`` checkpoints: a LLaMA backbone (saved under the ``model.`` prefix) plus a
linear ``score_head`` (hidden -> 1, with bias) read at the last non-padding token of
``BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}`` followed by the LLaMA EOS ``</s>``
(the Safe-RLHF training convention). No training RM/CM of any arm is used here.

Scoring runs over the SAME fixed batches as generation, so scores are identical for any shard layout.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .eval_common import EVALUATOR_TEMPLATE, fixed_batches, load_eval_prompts, run_dir, shard_file
from .modeling import autocast_ctx
from .utils import get_logger, load_config, pick_device, read_jsonl, write_json, write_jsonl

log = get_logger("saferl.eval_score")

PRIMARY_EVALUATORS = {"reward": "PKU-Alignment/beaver-7b-unified-reward", "cost": "PKU-Alignment/beaver-7b-unified-cost"}


def _resolve_file(repo_or_dir: str, filename: str, revision=None) -> Path:
    p = Path(repo_or_dir)
    if p.is_dir():
        return p / filename
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_or_dir, filename, revision=revision))


def load_score_head_weights(repo_or_dir: str, revision=None) -> Dict[str, torch.Tensor]:
    """Find and load ``score_head.weight`` / ``score_head.bias`` from a (sharded) safetensors checkpoint."""
    from safetensors import safe_open

    files: List[str]
    try:
        idx = json.loads(_resolve_file(repo_or_dir, "model.safetensors.index.json", revision).read_text())
        files = sorted({v for k, v in idx["weight_map"].items() if k.startswith("score_head.")})
    except Exception:
        files = ["model.safetensors"]
    out: Dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(str(_resolve_file(repo_or_dir, f, revision)), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if k.startswith("score_head."):
                    out[k] = sf.get_tensor(k)
    if "score_head.weight" not in out:
        raise ValueError(f"no score_head.weight found in {repo_or_dir}")
    return out


class PKUScoreModel(nn.Module):
    """LLaMA backbone + linear score head; score = head(h_last) at the last non-padding token."""

    def __init__(self, backbone: nn.Module, hidden_size: int, bias: bool, name: str, score_type: str):
        super().__init__()
        self.backbone = backbone
        self.score_head = nn.Linear(hidden_size, 1, bias=bias)
        self.name, self.score_type = name, score_type

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, device: str, dtype: torch.dtype = torch.bfloat16, revision=None) -> "PKUScoreModel":
        from transformers import AutoConfig, LlamaModel

        cfg = AutoConfig.from_pretrained(repo_or_dir, revision=revision)
        backbone = LlamaModel.from_pretrained(repo_or_dir, torch_dtype=dtype, revision=revision)  # "model." prefix stripped by HF
        head_sd = load_score_head_weights(repo_or_dir, revision)
        m = cls(backbone, cfg.hidden_size, "score_head.bias" in head_sd, repo_or_dir, getattr(cfg, "score_type", "unknown"))
        m.score_head.weight.data.copy_(head_sd["score_head.weight"].float())
        if "score_head.bias" in head_sd:
            m.score_head.bias.data.copy_(head_sd["score_head.bias"].float())
        m.score_head.to(dtype)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        m.to(device)
        return m

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        idx = attention_mask.long().sum(1) - 1
        last = h[torch.arange(h.shape[0], device=h.device), idx]
        return self.score_head(last).squeeze(-1).float()


def load_evaluator_tokenizer(repo_or_dir: str, revision=None):
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_or_dir, revision=revision)
    if tok.pad_token_id is None:
        cfg = AutoConfig.from_pretrained(repo_or_dir, revision=revision)
        tok.pad_token_id = int(cfg.pad_token_id) if cfg.pad_token_id is not None else tok.eos_token_id
    return tok


def encode_for_evaluator(tok, prompts: Sequence[str], responses: Sequence[str], template: str, append_eos: bool,
                         max_length: int) -> Dict[str, Any]:
    texts = [template.format(prompt=p, response=r) for p, r in zip(prompts, responses)]
    ids = [tok(t, add_special_tokens=True)["input_ids"] for t in texts]  # BOS added by the LLaMA tokenizer
    if append_eos:  # append the EOS *id* (the literal "</s>" string is not reliably parsed as the special token)
        ids = [x + [tok.eos_token_id] for x in ids]
    truncated = [len(x) > max_length for x in ids]
    ids = [x[:max_length] for x in ids]
    L = max(len(x) for x in ids)
    input_ids = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, :len(x)] = torch.tensor(x)
        attn[i, :len(x)] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "truncated": truncated, "lengths": [len(x) for x in ids]}


def scoring_chunks(manifest_indices: Sequence[int], batch_size: int, group_size: Optional[int]) -> List[List[int]]:
    """Positions (into the given lists) grouped into scoring batches that never cross a generation-batch boundary
    (generation batch = manifest_index // group_size). Batch composition is then a pure function of manifest position,
    so bf16 padding effects are identical for any shard layout."""
    chunks: List[List[int]] = []
    cur: List[int] = []
    cur_group = None
    for pos, mi in enumerate(manifest_indices):
        g = (mi // group_size) if group_size else 0
        if cur and (g != cur_group or len(cur) >= batch_size):
            chunks.append(cur); cur = []
        cur.append(pos); cur_group = g
    if cur:
        chunks.append(cur)
    return chunks


@torch.no_grad()
def score_pairs(model: PKUScoreModel, tok, prompts: Sequence[str], responses: Sequence[str], device: str, precision: str,
                template: str, append_eos: bool, max_length: int, batch_size: int,
                manifest_indices: Optional[Sequence[int]] = None, group_size: Optional[int] = None) -> Dict[str, List]:
    """Scores in fixed batches aligned to generation batches (no length sorting) so results do not depend on sharding."""
    scores: List[float] = [0.0] * len(prompts)
    truncated: List[bool] = [False] * len(prompts)
    mi = list(manifest_indices) if manifest_indices is not None else list(range(len(prompts)))
    for chunk in scoring_chunks(mi, batch_size, group_size):
        enc = encode_for_evaluator(tok, [prompts[i] for i in chunk], [responses[i] for i in chunk], template, append_eos, max_length)
        with autocast_ctx(device, precision):
            s = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for j, i in enumerate(chunk):
            scores[i] = float(s[j]); truncated[i] = bool(enc["truncated"][j])
    return {"scores": scores, "truncated": truncated}


def score_shard(cfg: Dict[str, Any], method: str, seed: int, num_shards: int, shard_id: int, overwrite: bool = False) -> Dict[str, Any]:
    d = run_dir(cfg, method, seed)
    in_path = shard_file(d, "responses", num_shards, shard_id)
    out_path = shard_file(d, "scores", num_shards, shard_id)
    if out_path.exists() and not overwrite:
        log.info("%s exists; skipping", out_path)
        return {"skipped": True, "file": str(out_path)}
    recs = read_jsonl(in_path)
    recs.sort(key=lambda r: r["manifest_index"])
    ev = cfg["evaluators"]
    device = pick_device(cfg.get("device", "auto"))
    prompts = [r["prompt"] for r in recs]
    responses = [r["response"] for r in recs]
    t0 = time.time()
    results: Dict[str, Dict[str, List]] = {}
    meta: Dict[str, Any] = {}
    for key in ("reward", "cost"):
        repo = ev[key]["model"]
        tok = load_evaluator_tokenizer(repo, ev[key].get("revision"))
        model = PKUScoreModel.from_pretrained(repo, device, torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32, ev[key].get("revision"))
        results[key] = score_pairs(model, tok, prompts, responses, device, cfg["precision"], ev.get("template", EVALUATOR_TEMPLATE),
                                   bool(ev.get("append_eos", True)), int(ev["max_length"]), int(ev["batch_size"]),
                                   manifest_indices=[r["manifest_index"] for r in recs], group_size=int(cfg["generation"]["batch_size"]))
        meta[key] = {"model": repo, "revision": ev[key].get("revision"), "score_type": model.score_type,
                     "n_parameters": sum(p.numel() for p in model.parameters())}
        log.info("%s scored %d responses with %s (%.0fs)", key, len(recs), repo, time.time() - t0)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    out = []
    for i, r in enumerate(recs):
        out.append({"prompt_hash": r["prompt_hash"], "manifest_index": r["manifest_index"], "method": method, "policy_seed": seed,
                    "reward_score": results["reward"]["scores"][i], "cost_score": results["cost"]["scores"][i],
                    "evaluator_input_truncated": bool(results["reward"]["truncated"][i] or results["cost"]["truncated"][i]),
                    "n_completion_tokens": r["n_completion_tokens"], "truncated": r["truncated"]})
    write_jsonl(out_path, out)
    summary = {"method": method, "policy_seed": seed, "num_shards": num_shards, "shard_id": shard_id, "n": len(out), "evaluators": meta,
               "template": ev.get("template", EVALUATOR_TEMPLATE), "append_eos": bool(ev.get("append_eos", True)), "max_length": ev["max_length"],
               "batch_size": ev["batch_size"], "precision": cfg["precision"], "device": device, "seconds": time.time() - t0,
               "mean_reward": sum(x["reward_score"] for x in out) / max(1, len(out)), "mean_cost": sum(x["cost_score"] for x in out) / max(1, len(out)),
               "file": str(out_path)}
    write_json(out_path.with_suffix(".summary.json"), summary)
    log.info("wrote %s: mean reward %.4f mean cost %.4f", out_path, summary["mean_reward"], summary["mean_cost"])
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    score_shard(cfg, args.method, args.seed, args.num_shards, args.shard_id, args.overwrite)


if __name__ == "__main__":
    main()
