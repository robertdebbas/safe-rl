"""Vanilla sigmoid DPO loss and masked sequence log-probabilities.

    L_DPO = -log sigmoid( beta * [ (log pi(y+|x) - log pi_ref(y+|x)) - (log pi(y-|x) - log pi_ref(y-|x)) ] )

``label_smoothing`` is implemented for completeness of the standard formula but is fixed to 0
in every config of this arm.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def sequence_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Sum of next-token log-probabilities over positions where ``loss_mask`` is 1.

    ``logits``   : [B, T, V] model outputs for ``input_ids`` [B, T]
    ``loss_mask``: [B, T] 1 for completion tokens (targets), 0 for prompt / padding.

    logits[:, t] predicts input_ids[:, t+1], so targets are input_ids[:, 1:] with mask
    loss_mask[:, 1:]. Prompt tokens therefore never contribute a term, but they condition every
    completion token through the model's hidden states.
    """
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(f"logits {tuple(logits.shape)} vs input_ids {tuple(input_ids.shape)}")
    targets = input_ids[:, 1:]
    mask = loss_mask[:, 1:].float()
    # Memory-light: per-row fp32 logsumexp + gather instead of materializing a full [B, T, V]
    # log-softmax copy (V ~ 150k for Qwen2.5). Numerically identical to log_softmax + gather.
    rows = []
    for b in range(logits.shape[0]):
        lg = logits[b, :-1, :].float()
        lse = torch.logsumexp(lg, dim=-1)
        tok = torch.gather(lg, dim=-1, index=targets[b].unsqueeze(-1)).squeeze(-1)
        rows.append(tok - lse)
    tok_logp = torch.stack(rows, dim=0) * mask
    return {
        "sum_logprob": tok_logp.sum(dim=-1),
        "num_tokens": mask.sum(dim=-1),
        "token_logprobs": tok_logp,
    }


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Per-example sigmoid DPO loss plus implicit rewards and margin.

    Returns dict with per-example tensors: loss, margin (= beta * logit), chosen_reward,
    rejected_reward.
    """
    chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps)
    margin = chosen_reward - rejected_reward
    loss = -F.logsigmoid(margin) * (1.0 - label_smoothing) - F.logsigmoid(-margin) * label_smoothing
    return {
        "loss": loss,
        "margin": margin,
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
    }
