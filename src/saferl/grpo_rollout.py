"""Rollout generation (Hugging Face ``generate`` on the policy under bf16 autocast) and per-token
log-probabilities of generated completions under any causal LM (policy, pi_old snapshot, pi_ref).

A trajectory = (prompt_ids, completion_ids). Prompt ids are the official Qwen chat serialization of
the user turn (identical to DPO/RM formatting); completion ids are exactly the sampled tokens up to and
including the terminating EOS (``<|im_end|>`` or ``<|endoftext|>``). A completion that reaches
``max_completion_length`` without EOS is flagged ``truncated`` (kept for training, reported).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .dpo_loss import sequence_logprobs
from .formatting import prompt_text
from .modeling import autocast_ctx


def eos_ids(tokenizer) -> List[int]:
    ids = [tokenizer.eos_token_id]
    extra = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if isinstance(extra, int) and extra >= 0 and extra not in ids:
        ids.append(extra)
    return ids


def encode_prompts_left_padded(tokenizer, prompts: Sequence[str], max_prompt_length: int, device: str) -> Dict[str, torch.Tensor]:
    texts = [prompt_text(tokenizer, p) for p in prompts]
    ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    for x in ids:
        if len(x) > max_prompt_length:
            raise ValueError(f"prompt has {len(x)} tokens > max_prompt_length={max_prompt_length}")
    L = max(len(x) for x in ids)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(ids), L), pad, dtype=torch.long)
    attn = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        input_ids[i, L - len(x):] = torch.tensor(x)
        attn[i, L - len(x):] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": attn.to(device), "prompt_ids": ids}


@torch.no_grad()
def generate_completions(model, tokenizer, prompts: Sequence[str], num_return_sequences: int, max_prompt_length: int,
                         max_completion_length: int, device: str, precision: str = "bf16", temperature: float = 1.0,
                         top_p: float = 1.0, repetition_penalty: float = 1.0, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Sample ``num_return_sequences`` completions per prompt with the CURRENT weights of ``model``
    (used as the frozen behavior policy for the rollout batch). Returns one record per trajectory,
    grouped by prompt (prompt-major order): prompt_index, k, prompt_ids, completion_ids, text, truncated."""
    was_training = model.training
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        elif device == "mps":
            torch.mps.manual_seed(seed)
    enc = encode_prompts_left_padded(tokenizer, prompts, max_prompt_length, device)
    stop = eos_ids(tokenizer)
    with autocast_ctx(device, precision):
        # NOTE: every sampling knob is set explicitly so the checkpoint's packaged generation_config cannot leak in
        # (Qwen2.5-Instruct ships top_k=20, temperature=0.7, top_p=0.8, repetition_penalty=1.1). top_k=0 disables top-k:
        # with temperature=1.0 and top_p=1.0 this is sampling from the full policy distribution, as pre-specified.
        out = model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], do_sample=True, temperature=temperature,
            top_p=top_p, top_k=0, repetition_penalty=repetition_penalty, max_new_tokens=max_completion_length,
            num_return_sequences=num_return_sequences, eos_token_id=stop, pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    L = enc["input_ids"].shape[1]
    gen = out[:, L:].cpu()
    stop_set = set(stop)
    records = []
    for row in range(gen.shape[0]):
        p_idx, k = divmod(row, num_return_sequences)
        toks = gen[row].tolist()
        comp, truncated = [], True
        for t in toks:
            comp.append(t)
            if t in stop_set:
                truncated = False
                break
        if truncated:
            comp = [t for t in comp if t != tokenizer.pad_token_id] if tokenizer.pad_token_id not in stop_set else comp
            comp = comp[:max_completion_length]
        text = tokenizer.decode([t for t in comp if t not in stop_set], skip_special_tokens=True)
        records.append({"prompt_index": p_idx, "k": k, "prompt_ids": enc["prompt_ids"][p_idx], "completion_ids": comp,
                        "text": text, "truncated": truncated, "n_completion_tokens": len(comp)})
    if was_training:
        model.train()
    return records


def collate_trajectories(trajs: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Right-padded prompt+completion sequences with ``loss_mask`` = 1 on completion tokens only."""
    seqs = [t["prompt_ids"] + t["completion_ids"] for t in trajs]
    L = max(len(s) for s in seqs)
    n = len(seqs)
    input_ids = torch.full((n, L), pad_token_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    loss_mask = torch.zeros((n, L), dtype=torch.long)
    for i, (t, s) in enumerate(zip(trajs, seqs)):
        input_ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
        loss_mask[i, len(t["prompt_ids"]):len(s)] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "loss_mask": loss_mask}


def token_logprobs(model, batch: Dict[str, torch.Tensor], device: str, precision: str) -> Dict[str, torch.Tensor]:
    """Per-token log-probs [B, T-1] of input_ids[:, 1:] under ``model`` plus the aligned mask [B, T-1]
    (1 on generated completion tokens). Differentiable w.r.t. ``model`` if grad is enabled."""
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    with autocast_ctx(device, precision):
        logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
    out = sequence_logprobs(logits, input_ids, loss_mask)
    return {"token_logprobs": out["token_logprobs"], "mask": loss_mask[:, 1:].float(), "sum_logprob": out["sum_logprob"],
            "num_tokens": out["num_tokens"]}


@torch.no_grad()
def batched_token_logprobs(model, trajs: Sequence[Dict[str, Any]], pad_token_id: int, micro_batch: int, device: str,
                           precision: str) -> List[torch.Tensor]:
    """No-grad per-trajectory tensors of completion-token log-probs (variable length, CPU float32)."""
    was_training = model.training
    model.eval()
    out: List[torch.Tensor] = []
    for start in range(0, len(trajs), micro_batch):
        chunk = trajs[start:start + micro_batch]
        b = collate_trajectories(chunk, pad_token_id)
        r = token_logprobs(model, b, device, precision)
        for i, t in enumerate(chunk):
            p = len(t["prompt_ids"])
            out.append(r["token_logprobs"][i, p - 1: p - 1 + len(t["completion_ids"])].float().cpu())
    if was_training:
        model.train()
    return out
