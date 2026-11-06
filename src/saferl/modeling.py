"""Model loading, mixed precision, batched sequence log-probs, and the reference log-prob cache."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .dpo_loss import sequence_logprobs
from .formatting import collate_dpo, encode_dpo_row
from .utils import get_logger, read_json, sha256_text, write_json

log = get_logger("saferl.modeling")


def autocast_ctx(device: str, precision: str):
    """bf16 autocast around forward/backward; parameters (master weights) stay fp32."""
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "bf16":
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {precision!r}")
    device_type = device.split(":")[0]
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        raise ValueError("tokenizer has no pad token")
    return tok


def load_model(name: str, device: str, dtype: torch.dtype = torch.float32, gradient_checkpointing: bool = False,
               attn_implementation: Optional[str] = None):
    from transformers import AutoModelForCausalLM

    kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    return model


def encode_rows(tokenizer, rows: Sequence[Dict[str, Any]], max_prompt_length: Optional[int], max_length: Optional[int],
                on_overflow: str = "error") -> List[Dict[str, Any]]:
    return [encode_dpo_row(tokenizer, r, max_prompt_length, max_length, on_overflow) for r in rows]


def forward_logprobs(model, batch: Dict[str, Any], device: str, precision: str) -> Dict[str, torch.Tensor]:
    """One forward over a collated batch (chosen rows then rejected rows). Returns per-row
    chosen/rejected summed completion log-probs (float32) and completion token counts."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    with autocast_ctx(device, precision):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    out = sequence_logprobs(logits, input_ids, loss_mask)
    B = batch["batch_size"]
    return {
        "chosen_logps": out["sum_logprob"][:B],
        "rejected_logps": out["sum_logprob"][B:],
        "chosen_tokens": out["num_tokens"][:B],
        "rejected_tokens": out["num_tokens"][B:],
    }


@torch.no_grad()
def compute_logprobs(model, encoded_rows: Sequence[Dict[str, Any]], pad_token_id: int, batch_size: int,
                     device: str, precision: str, desc: str = "logprobs") -> Dict[str, Dict[str, float]]:
    """Sequence log-probs for every row (no grad, eval mode). Restores the previous train/eval mode."""
    was_training = model.training
    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    # Sort by length for efficient padding; results are keyed by row_id so order is irrelevant.
    order = sorted(range(len(encoded_rows)), key=lambda i: max(len(encoded_rows[i]["chosen"]["input_ids"]),
                                                               len(encoded_rows[i]["rejected"]["input_ids"])))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = collate_dpo([encoded_rows[i] for i in idx], pad_token_id)
        out = forward_logprobs(model, batch, device, precision)
        for j, rid in enumerate(batch["row_ids"]):
            results[rid] = {
                "chosen": float(out["chosen_logps"][j]),
                "rejected": float(out["rejected_logps"][j]),
                "chosen_tokens": int(out["chosen_tokens"][j]),
                "rejected_tokens": int(out["rejected_tokens"][j]),
            }
        if (start // batch_size) % 50 == 0:
            log.info("%s: %d/%d rows", desc, min(start + batch_size, len(order)), len(order))
    if was_training:
        model.train()
    return results


# --------------------------------------------------------------------------- reference cache

def ref_cache_key(model_name: str, rows_sha256: str, max_prompt_length: int, max_length: int,
                  device: str, precision: str) -> str:
    payload = json.dumps({
        "model": model_name, "rows_sha256": rows_sha256, "max_prompt_length": max_prompt_length,
        "max_length": max_length, "device_type": device.split(":")[0], "precision": precision,
    }, sort_keys=True)
    return sha256_text(payload)[:16]


def get_or_compute_ref_logprobs(cache_dir: Optional[str], split: str, key: str, compute_fn) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """Returns (logprobs, loaded_from_cache). ``compute_fn`` is called only on a cache miss."""
    if cache_dir is not None:
        path = Path(cache_dir) / f"{split}.{key}.json"
        if path.exists():
            log.info("reference log-probs for %s loaded from cache %s", split, path)
            return read_json(path), True
    res = compute_fn()
    if cache_dir is not None:
        write_json(path, res)
        log.info("reference log-probs for %s cached at %s", split, path)
    return res, False


def save_checkpoint(model, tokenizer, path: Path, dtype: str, state: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    # NOTE: move to CPU first, then cast. A fused ``.to("cpu", dtype)`` on an MPS tensor was observed
    # (torch 2.8, macOS) to corrupt the SOURCE parameters in place; two separate ops are safe.
    sd = {k: v.detach().cpu().to(torch_dtype).clone() for k, v in model.state_dict().items()}
    model.save_pretrained(path, state_dict=sd, safe_serialization=True)
    tokenizer.save_pretrained(path)
    write_json(path / "trainer_state.json", state)
    del sd
