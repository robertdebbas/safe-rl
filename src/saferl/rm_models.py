"""Score models: a Qwen2.5 backbone (no LM head) plus one or more learned linear scalar heads.

score_head(x, y) = w^T h_last + b, where h_last is the final hidden state (after the model's final
norm) at the LAST NON-PADDING token of the serialized prompt + response + <|im_end|> sequence,
i.e. at the appended EOS token. Prompt tokens are context only; there is no token-level loss.

Heads: {"scalar"} (Architecture A), {"reward", "cost"} on one shared backbone (Architecture B),
{"reward"} or {"cost"} alone (Architecture C). Larger reward/scalar = more preferred/helpful;
larger cost = more harmful, with C = 0 as the learned safe/unsafe boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .formatting import encode_example, pad_batch
from .modeling import autocast_ctx
from .utils import get_logger, write_json

log = get_logger("saferl.rm_models")

CONFIG_NAME = "score_model_config.json"
HEADS_NAME = "heads.pt"


def last_token_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state at the last non-padding position of each (right-padded) sequence."""
    idx = attention_mask.long().sum(dim=1) - 1  # [B]
    if (idx < 0).any():
        raise ValueError("empty sequence in batch")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]


class ScoreModel(nn.Module):
    def __init__(self, backbone: nn.Module, head_names: Sequence[str], hidden_size: int, init_std: float,
                 backbone_name: str = ""):
        super().__init__()
        self.backbone = backbone
        self.head_names = list(head_names)
        self.backbone_name = backbone_name
        self.heads = nn.ModuleDict({n: nn.Linear(hidden_size, 1) for n in self.head_names})
        for h in self.heads.values():  # backbone's standard initializer (normal, initializer_range), zero bias
            nn.init.normal_(h.weight, mean=0.0, std=init_std)
            nn.init.zeros_(h.bias)

    # ------------------------------------------------------------------ construction / io
    @classmethod
    def from_backbone(cls, name: str, head_names: Sequence[str], device: str, dtype: torch.dtype = torch.float32,
                      gradient_checkpointing: bool = False) -> "ScoreModel":
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(name, torch_dtype=dtype)
        backbone.config.use_cache = False
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m = cls(backbone, head_names, backbone.config.hidden_size, float(backbone.config.initializer_range), name)
        m.to(device)
        return m

    def save_pretrained(self, path: Path, dtype: str = "bfloat16") -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        # two-step cast (cpu, then dtype): see modeling.save_checkpoint
        sd = {k: v.detach().cpu().to(torch_dtype).clone() for k, v in self.backbone.state_dict().items()}
        self.backbone.save_pretrained(path, state_dict=sd, safe_serialization=True)
        heads_sd = {k: v.detach().cpu().float().clone() for k, v in self.heads.state_dict().items()}  # heads always fp32
        torch.save(heads_sd, path / HEADS_NAME)
        write_json(path / CONFIG_NAME, {"head_names": self.head_names, "backbone_name": self.backbone_name,
                                         "hidden_size": self.heads[self.head_names[0]].in_features, "backbone_dtype": dtype})
        del sd, heads_sd

    @classmethod
    def load(cls, path: str, device: str, dtype: torch.dtype = torch.float32) -> "ScoreModel":
        from transformers import AutoModel

        path = Path(path)
        cfg = json.loads((path / CONFIG_NAME).read_text())
        backbone = AutoModel.from_pretrained(path, torch_dtype=dtype)
        backbone.config.use_cache = False
        m = cls(backbone, cfg["head_names"], cfg["hidden_size"], float(backbone.config.initializer_range), cfg["backbone_name"])
        m.heads.load_state_dict(torch.load(path / HEADS_NAME, map_location="cpu"))
        m.to(device)
        return m

    # ------------------------------------------------------------------ forward
    def hidden_last(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return last_token_hidden(out.last_hidden_state, attention_mask)

    def apply_heads(self, h_last: torch.Tensor, names: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        """Scalar scores per head, computed in fp32 regardless of any enclosing autocast."""
        names = self.head_names if names is None else names
        device_type = h_last.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            h = h_last.float()
            return {n: self.heads[n](h).squeeze(-1) for n in names}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                heads: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        return self.apply_heads(self.hidden_last(input_ids, attention_mask), heads)

    def num_parameters(self) -> Dict[str, int]:
        bb = sum(p.numel() for p in self.backbone.parameters())
        heads = {n: sum(p.numel() for p in h.parameters()) for n, h in self.heads.items()}
        return {"backbone": bb, "heads": heads, "total": bb + sum(heads.values())}

    # ------------------------------------------------------------------ batched inference interface
    @torch.no_grad()
    def score(self, tokenizer, prompts: Sequence[str], responses: Sequence[str], head: str, batch_size: int = 8,
              max_prompt_length: Optional[int] = None, max_length: Optional[int] = None, precision: str = "bf16",
              on_overflow: str = "truncate") -> List[float]:
        """Batched scores for (prompt, response) pairs under ``head``. Same formatting as training."""
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        if head not in self.heads:
            raise KeyError(f"head {head!r} not in {self.head_names}")
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        encs = [encode_example(tokenizer, p, r, max_prompt_length, max_length, on_overflow) for p, r in zip(prompts, responses)]
        out = [0.0] * len(encs)
        order = sorted(range(len(encs)), key=lambda i: len(encs[i]["input_ids"]))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            b = pad_batch([encs[i] for i in idx], tokenizer.pad_token_id)
            with autocast_ctx(str(device), precision):
                h = self.hidden_last(b["input_ids"].to(device), b["attention_mask"].to(device))
            s = self.apply_heads(h, [head])[head]
            for j, i in enumerate(idx):
                out[i] = float(s[j])
        if was_training:
            self.train()
        return out

    def score_reward(self, tokenizer, prompts, responses, **kw) -> List[float]:
        return self.score(tokenizer, prompts, responses, "reward", **kw)

    def score_cost(self, tokenizer, prompts, responses, **kw) -> List[float]:
        return self.score(tokenizer, prompts, responses, "cost", **kw)

    def score_scalar(self, tokenizer, prompts, responses, **kw) -> List[float]:
        return self.score(tokenizer, prompts, responses, "scalar", **kw)


def decay_param_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """AdamW groups: decay on >=2-D weights (linear/embedding matrices, head weight); no decay on
    biases and normalization weights."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
