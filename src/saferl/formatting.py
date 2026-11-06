"""Chat formatting + tokenization for DPO rows.

The policy sees exactly the official Qwen2.5-Instruct chat template applied to a single
user turn (no custom system prompt, no objective text), followed by the response and EOS.

    prompt_text   = apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    completion    = response + eos_token            # eos = "<|im_end|>" for Qwen2.5-Instruct

Prompt tokens are CONTEXT ONLY: ``loss_mask`` is 0 on prompt positions and 1 on completion
positions (including EOS), so the DPO sequence log-probability is sum_t log pi(y_t | x, y_<t).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

FORBIDDEN_IN_PROMPT = (
    "objective", "helpfulness", "harmlessness", "harmless", "be helpful", "safety",
)


def prompt_text(tokenizer, prompt: str) -> str:
    """Serialized prompt via the official chat template, ending in the assistant header."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def completion_text(tokenizer, response: str) -> str:
    return response + tokenizer.eos_token


def full_text(tokenizer, prompt: str, response: str) -> str:
    return prompt_text(tokenizer, prompt) + completion_text(tokenizer, response)


def tokenize_prompt(tokenizer, prompt: str) -> List[int]:
    # The template already contains all special tokens; do not let the tokenizer add BOS etc.
    return tokenizer(prompt_text(tokenizer, prompt), add_special_tokens=False)["input_ids"]


def tokenize_completion(tokenizer, response: str) -> List[int]:
    ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    return ids + [tokenizer.eos_token_id]


def encode_example(
    tokenizer,
    prompt: str,
    response: str,
    max_prompt_length: Optional[int] = None,
    max_length: Optional[int] = None,
    on_overflow: str = "error",
) -> Dict[str, Any]:
    """Return input_ids / attention_mask / loss_mask for one (prompt, response).

    ``on_overflow``: 'error' (default; the data pipeline already excluded over-length pairs so
    an overflow here is a bug), or 'truncate' (prompt left-truncated to max_prompt_length, then
    completion right-truncated to fit max_length; the EOS is dropped in that case and the
    example is flagged ``truncated=True``).
    """
    p_ids = tokenize_prompt(tokenizer, prompt)
    c_ids = tokenize_completion(tokenizer, response)
    truncated = False
    if max_prompt_length is not None and len(p_ids) > max_prompt_length:
        if on_overflow == "error":
            raise ValueError(f"prompt has {len(p_ids)} tokens > max_prompt_length={max_prompt_length}")
        p_ids = p_ids[-max_prompt_length:]
        truncated = True
    if max_length is not None and len(p_ids) + len(c_ids) > max_length:
        if on_overflow == "error":
            raise ValueError(
                f"sequence has {len(p_ids) + len(c_ids)} tokens > max_length={max_length}")
        c_ids = c_ids[: max(0, max_length - len(p_ids))]
        truncated = True
    ids = p_ids + c_ids
    loss_mask = [0] * len(p_ids) + [1] * len(c_ids)
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "loss_mask": loss_mask,
        "prompt_len": len(p_ids),
        "completion_len": len(c_ids),
        "truncated": truncated,
    }


def encode_dpo_row(tokenizer, row: Dict[str, Any], max_prompt_length=None, max_length=None,
                   on_overflow: str = "error") -> Dict[str, Any]:
    """Encode chosen and rejected with an identical prompt serialization."""
    chosen = encode_example(tokenizer, row["prompt"], row["chosen"], max_prompt_length, max_length, on_overflow)
    rejected = encode_example(tokenizer, row["prompt"], row["rejected"], max_prompt_length, max_length, on_overflow)
    assert chosen["input_ids"][: chosen["prompt_len"]] == rejected["input_ids"][: rejected["prompt_len"]]
    return {"row_id": row["row_id"], "chosen": chosen, "rejected": rejected}


def pad_batch(examples: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Right-pad a list of encoded examples (each with input_ids/attention_mask/loss_mask)."""
    max_len = max(len(e["input_ids"]) for e in examples)
    n = len(examples)
    input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)
    loss_mask = torch.zeros((n, max_len), dtype=torch.long)
    for i, e in enumerate(examples):
        L = len(e["input_ids"])
        input_ids[i, :L] = torch.tensor(e["input_ids"], dtype=torch.long)
        attention_mask[i, :L] = 1
        loss_mask[i, :L] = torch.tensor(e["loss_mask"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}


def collate_dpo(encoded_rows: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """Concatenate chosen then rejected sequences into one padded batch of size 2B.

    Rows [0, B) are chosen, rows [B, 2B) are rejected, aligned by index.
    """
    chosen = [r["chosen"] for r in encoded_rows]
    rejected = [r["rejected"] for r in encoded_rows]
    batch = pad_batch(list(chosen) + list(rejected), pad_token_id)
    batch["row_ids"] = [r["row_id"] for r in encoded_rows]
    batch["batch_size"] = len(encoded_rows)
    return batch


def serialize_for_inspection(tokenizer, row: Dict[str, Any]) -> str:
    """Human-readable dump of exactly what the model sees for one DPO row."""
    enc = encode_dpo_row(tokenizer, row, on_overflow="truncate")
    lines = [
        f"row_id      : {row['row_id']}",
        f"objective   : {row['objective']}   (metadata only; NOT in model input)",
        f"pair_type   : {row['pair_type']}",
        f"chosen_id   : response_{row['chosen_id']}   rejected_id: response_{row['rejected_id']}",
        "",
        "=== CHOSEN sequence (prompt tokens masked from the loss) ===",
        full_text(tokenizer, row["prompt"], row["chosen"]),
        f"[prompt_len={enc['chosen']['prompt_len']} completion_len={enc['chosen']['completion_len']}]",
        "",
        "=== REJECTED sequence ===",
        full_text(tokenizer, row["prompt"], row["rejected"]),
        f"[prompt_len={enc['rejected']['prompt_len']} completion_len={enc['rejected']['completion_len']}]",
        "",
        "=== prompt token ids (first 32) ===",
        str(enc["chosen"]["input_ids"][: min(32, enc["chosen"]["prompt_len"])]),
        "=== loss_mask boundary (last 3 prompt positions, first 3 completion positions) ===",
        str(enc["chosen"]["loss_mask"][enc["chosen"]["prompt_len"] - 3: enc["chosen"]["prompt_len"] + 3]),
    ]
    return "\n".join(lines)


def assert_no_objective_leakage(tokenizer, row: Dict[str, Any]) -> None:
    """The serialized prompt must be a pure function of the user prompt (and the official
    template); it must not contain any objective text that is not already in the user prompt."""
    serialized = prompt_text(tokenizer, row["prompt"])
    user_part = row["prompt"]
    template_only = serialized.replace(user_part, "", 1).lower()
    for token in FORBIDDEN_IN_PROMPT:
        if token in template_only:
            raise AssertionError(f"objective-related text {token!r} found in template scaffolding")
    if row["objective"].lower() in template_only:
        raise AssertionError("objective name leaked into the serialized prompt")
