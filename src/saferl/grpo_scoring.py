"""Frozen reward / cost scorers for the RL stage, built from trained ScoreModel checkpoints.

Arms:
  scalar   : one ScoreModel with head "scalar"            -> scores {"scalar": F}
  shared   : one ScoreModel with heads "reward", "cost"   -> scores {"reward": R, "cost": C}
  separate : two ScoreModels, heads "reward" and "cost"   -> scores {"reward": R, "cost": C}

All scorer parameters are frozen (requires_grad False, eval mode, no grad). Scores are RAW model
outputs: for the primary pairwise-trained cost models, C is identified only up to an additive
constant, so nothing here (or downstream) interprets C = 0 as a safety boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch

from .rm_models import ScoreModel
from .utils import sha256_file

ARM_HEADS = {"scalar": ["scalar"], "shared": ["reward", "cost"], "separate": ["reward", "cost"]}


def checkpoint_fingerprint(path: str) -> Dict[str, str]:
    p = Path(path).resolve()
    out = {"path": str(p)}
    for f in ("model.safetensors", "heads.pt"):
        if (p / f).exists():
            out[f] = sha256_file(p / f)
    return out


class FrozenScorers:
    def __init__(self, arm: str, checkpoints: Dict[str, str], device: str, precision: str = "bf16", batch_size: int = 16,
                 max_prompt_length: int = 512, max_length: int = 1024):
        """``checkpoints``: {"scalar": path} | {"shared": path} | {"reward": path, "cost": path}."""
        if arm not in ARM_HEADS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm, self.device, self.precision, self.batch_size = arm, device, precision, batch_size
        self.max_prompt_length, self.max_length = max_prompt_length, max_length
        self.models: Dict[str, ScoreModel] = {}   # head name -> model providing it
        self.fingerprints: Dict[str, Dict[str, str]] = {}
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        if arm == "scalar":
            m = self._load(checkpoints["scalar"], dtype)
            self.models["scalar"] = m
            self.fingerprints["scalar"] = checkpoint_fingerprint(checkpoints["scalar"])
        elif arm == "shared":
            m = self._load(checkpoints["shared"], dtype)
            self.models["reward"] = m
            self.models["cost"] = m
            self.fingerprints["shared"] = checkpoint_fingerprint(checkpoints["shared"])
        else:
            self.models["reward"] = self._load(checkpoints["reward"], dtype)
            self.models["cost"] = self._load(checkpoints["cost"], dtype)
            self.fingerprints["reward"] = checkpoint_fingerprint(checkpoints["reward"])
            self.fingerprints["cost"] = checkpoint_fingerprint(checkpoints["cost"])
        for head, m in self.models.items():
            if head not in m.head_names:
                raise ValueError(f"checkpoint for arm {arm!r} lacks head {head!r} (has {m.head_names})")

    def _load(self, path: str, dtype: torch.dtype) -> ScoreModel:
        m = ScoreModel.load(path, self.device, dtype)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    @property
    def heads(self) -> List[str]:
        return ARM_HEADS[self.arm]

    @torch.no_grad()
    def score(self, tokenizer, prompts: Sequence[str], responses: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Raw scores per head for aligned (prompt, response) lists; shared backbone is run once."""
        out: Dict[str, torch.Tensor] = {}
        done: Dict[int, Dict[str, List[float]]] = {}
        for head in self.heads:
            m = self.models[head]
            key = id(m)
            if key not in done:
                done[key] = self._score_all_heads(m, tokenizer, prompts, responses)
            out[head] = torch.tensor(done[key][head], dtype=torch.float32)
        return out

    @torch.no_grad()
    def _score_all_heads(self, m: ScoreModel, tokenizer, prompts, responses) -> Dict[str, List[float]]:
        from .formatting import encode_example, pad_batch
        from .modeling import autocast_ctx

        encs = [encode_example(tokenizer, p, r, self.max_prompt_length, self.max_length, "truncate") for p, r in zip(prompts, responses)]
        res = {h: [0.0] * len(encs) for h in m.head_names}
        order = sorted(range(len(encs)), key=lambda i: len(encs[i]["input_ids"]))
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            b = pad_batch([encs[i] for i in idx], tokenizer.pad_token_id)
            with autocast_ctx(self.device, self.precision):
                h = m.hidden_last(b["input_ids"].to(self.device), b["attention_mask"].to(self.device))
            sc = m.apply_heads(h)
            for j, i in enumerate(idx):
                for hn in m.head_names:
                    res[hn][i] = float(sc[hn][j])
        return res

    def num_parameters(self) -> int:
        seen, total = set(), 0
        for m in self.models.values():
            if id(m) not in seen:
                seen.add(id(m))
                total += m.num_parameters()["total"]
        return total
