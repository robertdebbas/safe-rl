"""Reward / cost model losses.

Bradley-Terry (helpfulness, or scalar overall):   L_BT = -log sigmoid(F(x,y+) - F(x,y-))
Cost ranking (larger C = more harmful):           L_C = -log sigmoid(C(x,y_harmful) - C(x,y_safer))
                                                  y_safer selected by safer_response_id  (PRIMARY: pairwise only)
Optional ablation (cost_pointwise_classification = true; OFF in all primary configs):
  Safe-RLHF-style anchored cost  L_C = L_C_rank - log sigmoid(s(y0) C(x,y0)) - log sigmoid(s(y1) C(x,y1)),
  s(y) = +1 if unsafe, -1 if safe (PKU pointwise labels), which pins the safe/unsafe boundary at C = 0.
Pairwise BT identifies C only up to an additive constant; downstream constraint thresholds must be
calibrated from the frozen cost model (e.g. mean cost of the initial policy), never hard-coded to 0.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def bt_loss(score_pos: torch.Tensor, score_neg: torch.Tensor) -> Dict[str, torch.Tensor]:
    margin = score_pos - score_neg
    return {"loss": -F.logsigmoid(margin), "margin": margin}


def cost_class_loss(cost: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Per-response pointwise loss; ``sign`` is +1 (unsafe) / -1 (safe)."""
    return -F.logsigmoid(sign.to(cost.dtype) * cost)


def cost_loss(cost_safe: torch.Tensor, cost_harmful: torch.Tensor, sign_safe: torch.Tensor = None,
              sign_harmful: torch.Tensor = None, pointwise_classification: bool = False) -> Dict[str, torch.Tensor]:
    """``cost_safe`` scores y_safer (the safer_response_id response), ``cost_harmful`` the other one.
    PRIMARY (``pointwise_classification=False``): pairwise BT only; ``sign_*`` are ignored.
    Ablation (``True``): adds the pointwise terms using ``sign_*`` (+1 unsafe / -1 safe)."""
    margin = cost_harmful - cost_safe  # positive = ranking correct
    rank = -F.logsigmoid(margin)
    if pointwise_classification:
        if sign_safe is None or sign_harmful is None:
            raise ValueError("pointwise classification requires sign_safe / sign_harmful")
        cls = cost_class_loss(cost_safe, sign_safe) + cost_class_loss(cost_harmful, sign_harmful)
    else:
        cls = torch.zeros_like(rank)
    return {"loss": rank + cls, "rank_loss": rank, "class_loss": cls, "margin": margin}
