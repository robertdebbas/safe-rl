"""Pure-math core of (constrained) GRPO: group advantages, Lagrangian scores, dual update, and the
clipped token objective with the reference KL. No model code here, so every piece is unit-testable.

Ordinary GRPO (scalar RM):        A_i = (r_i - mean_group r) / (std_group r + eps)
Constrained GRPO (R, C, lambda), revision of 2026-11-01:
   A_R = (R - mean_group R) / (std_group R + eps),  A_C = (C - mean_group C) / (std_group C + eps),  A = A_R - lambda * A_C
   (each component normalized separately within its prompt group; population std, unbiased=False; the earlier joint
   normalization of Z = R - lambda C is no longer used)
Zero-variance group / component:  that advantage (component) is 0 (counted)
Dual update (once per rollout batch, after the policy updates of that batch):
   C_ema = C_batch (first batch) else 0.9 C_ema + 0.1 C_batch
   mu <- mu + eta * lambda * (C_ema - d) ;  lambda = exp(mu)        (lambda_0 = 1, eta = 0.01)
Policy objective per generated token (prompt and padding masked):
   rho = exp(logp_new - logp_old);  surrogate = min(rho A, clip(rho, 1-eps, 1+eps) A)
   KL  = exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1     (non-negative sampled estimator)
   J   = mean_traj [ (1/|o_i|) sum_t ( surrogate_it - beta_KL * KL_it ) ] ;  loss = -J
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


# --------------------------------------------------------------------------- advantages

def group_advantages(scores: torch.Tensor, group_size: int, eps: float = 1e-6) -> Dict[str, Any]:
    """``scores``: [N] with N = n_groups * group_size, trajectories of one prompt contiguous.
    Returns detached advantages [N] and diagnostics."""
    scores = scores.detach().float()
    if scores.numel() % group_size != 0:
        raise ValueError(f"{scores.numel()} scores not divisible by group_size={group_size}")
    g = scores.view(-1, group_size)
    mean = g.mean(dim=1, keepdim=True)
    std = g.std(dim=1, keepdim=True, unbiased=False)          # population std
    zero_var = std <= 1e-8                                      # effectively zero -> zero advantage for the group
    adv = (g - mean) / (std + eps)
    adv = torch.where(zero_var.expand_as(adv), torch.zeros_like(adv), adv)
    return {"advantages": adv.reshape(-1).detach(), "n_groups": int(g.shape[0]),
            "n_zero_variance_groups": int(zero_var.sum()), "group_mean": mean.squeeze(1), "group_std": std.squeeze(1)}


def lagrangian_scores(reward: torch.Tensor, cost: torch.Tensor, lam: float) -> torch.Tensor:
    """Z = R - lambda * C on the RAW scores (logged for diagnostics only since the 2026-11-01 revision)."""
    return reward.detach().float() - float(lam) * cost.detach().float()


def constrained_advantages(reward: torch.Tensor, cost: torch.Tensor, lam: float, group_size: int, eps: float = 1e-6) -> Dict[str, Any]:
    """A = A_R - lambda * A_C with A_R, A_C each group-normalized separately (population std + eps)."""
    ar = group_advantages(reward, group_size, eps)
    ac = group_advantages(cost, group_size, eps)
    adv = (ar["advantages"] - float(lam) * ac["advantages"]).detach()
    return {"advantages": adv, "adv_reward": ar["advantages"], "adv_cost": ac["advantages"], "n_groups": ar["n_groups"],
            "n_zero_variance_groups": ar["n_zero_variance_groups"] + ac["n_zero_variance_groups"],
            "n_zero_variance_reward_groups": ar["n_zero_variance_groups"], "n_zero_variance_cost_groups": ac["n_zero_variance_groups"]}


# --------------------------------------------------------------------------- dual variable

@dataclass
class DualState:
    d: float                       # calibrated threshold (mean initial-policy cost under this arm's cost model)
    eta: float = 0.01
    log_lambda: float = 0.0        # mu = log(lambda); lambda_0 = 1
    ema: Optional[float] = None    # cost EMA (None until the first rollout batch)
    ema_decay: float = 0.9
    n_updates: int = 0
    history: list = field(default_factory=list)

    @property
    def lam(self) -> float:
        import math
        return math.exp(self.log_lambda)

    def update(self, c_batch: float) -> Dict[str, float]:
        """One dual step after a rollout batch: EMA update then mu += eta * lambda_k * (C_ema - d)."""
        import math
        c_batch = float(c_batch)
        self.ema = c_batch if self.ema is None else self.ema_decay * self.ema + (1.0 - self.ema_decay) * c_batch
        lam_k = self.lam
        self.log_lambda = self.log_lambda + self.eta * lam_k * (self.ema - self.d)
        self.n_updates += 1
        rec = {"lambda_prev": lam_k, "lambda": math.exp(self.log_lambda), "log_lambda": self.log_lambda,
               "c_batch": c_batch, "c_ema": self.ema, "d": self.d, "violation": self.ema - self.d}
        self.history.append(rec)
        return rec

    def state_dict(self) -> Dict[str, Any]:
        return {"d": self.d, "eta": self.eta, "log_lambda": self.log_lambda, "ema": self.ema, "ema_decay": self.ema_decay,
                "n_updates": self.n_updates}

    @classmethod
    def from_state_dict(cls, sd: Dict[str, Any]) -> "DualState":
        return cls(d=sd["d"], eta=sd["eta"], log_lambda=sd["log_lambda"], ema=sd["ema"], ema_decay=sd["ema_decay"],
                   n_updates=sd["n_updates"])


# --------------------------------------------------------------------------- policy objective

def grpo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, logp_ref: torch.Tensor, advantages: torch.Tensor,
              mask: torch.Tensor, clip_eps: float = 0.2, beta_kl: float = 0.02) -> Dict[str, torch.Tensor]:
    """Per-token tensors [B, T] (log-probs of the generated tokens; ``mask`` 1 on generated completion
    tokens, 0 on prompt/padding). ``advantages`` [B] (constants). Returns the minimized loss (mean over
    trajectories of the per-trajectory token mean) plus diagnostics."""
    adv = advantages.detach().float().unsqueeze(1)
    mask = mask.float()
    logp_old = logp_old.detach().float()
    logp_ref = logp_ref.detach().float()
    logp_new = logp_new.float()
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)
    delta = logp_ref - logp_new
    kl = torch.exp(delta) - delta - 1.0
    per_token = surrogate - beta_kl * kl
    n_tok = mask.sum(dim=1).clamp(min=1.0)
    per_traj = (per_token * mask).sum(dim=1) / n_tok
    loss = -per_traj.mean()
    with torch.no_grad():
        tot = mask.sum().clamp(min=1.0)
        clipped_frac = (((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float() * mask).sum() / tot
        stats = {
            "loss_value": loss.detach(), "surrogate_mean": (surrogate * mask).sum() / tot, "kl_mean": (kl * mask).sum() / tot,
            "ratio_mean": (ratio * mask).sum() / tot, "clip_fraction": clipped_frac, "n_tokens": mask.sum(),
            "per_traj_objective": per_traj.detach(),
        }
    return {"loss": loss, **stats}
