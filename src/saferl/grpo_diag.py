"""Collapse diagnostics and gates for GRPO policies under FULL-distribution sampling.

Measures, on a fixed set of sampled responses:
  * tail mass: mean over generated tokens of the policy's probability mass outside its own top-k (k = 20);
  * entropy of the next-token distribution;
  * coherence proxy: fraction of ASCII characters in the decoded text (PKU prompts are English);
  * truncation rate, completion length, sampled KL to the reference, finiteness of training records.
Gates are plain threshold checks returned as (passed, details) so they can be unit-tested.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .grpo_rollout import collate_trajectories
from .modeling import autocast_ctx

DEFAULT_GATES: Dict[str, float] = {}       # revision 2026-11-01: NO numerical gate; automatic decision = hard failures only (non-finite / OOM)
DEFAULT_COLLAPSE: Dict[str, float] = {}    # continuation auto-stop: only a clear catastrophic failure (non-finite values)
DIAGNOSTIC_ONLY = ("tail_mass", "kl_mean", "truncation_rate", "ascii_mean", "ascii_low_frac", "entropy", "completion_len_mean")  # reported, never gating


@torch.no_grad()
def tail_mass_and_entropy(model, trajs: Sequence[Dict[str, Any]], pad_token_id: int, device: str, precision: str,
                          k: int = 20, micro_batch: int = 8) -> Dict[str, float]:
    """Teacher-forced pass over completion positions of ``trajs``: mean mass outside the policy's top-k and entropy."""
    was_training = model.training
    model.eval()
    tail, ent, n = 0.0, 0.0, 0
    finite = True
    for start in range(0, len(trajs), micro_batch):
        chunk = trajs[start:start + micro_batch]
        b = collate_trajectories(chunk, pad_token_id)
        with autocast_ctx(device, precision):
            logits = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device), use_cache=False).logits
        probs = torch.softmax(logits[:, :-1].float(), dim=-1)
        mask = b["loss_mask"][:, 1:].to(device).bool()
        if not torch.isfinite(probs).all():
            finite = False
        topk = torch.topk(probs, k, dim=-1).values.sum(-1)
        e = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
        tail += float((1.0 - topk)[mask].sum()); ent += float(e[mask].sum()); n += int(mask.sum())
    if was_training:
        model.train()
    return {"tail_mass": tail / max(1, n), "entropy": ent / max(1, n), "n_tokens": n, "finite": finite, "k": k}


def ascii_stats(texts: Sequence[str]) -> Dict[str, float]:
    fr = [sum(c.isascii() for c in t) / max(1, len(t)) for t in texts]
    return {"ascii_mean": float(np.mean(fr)) if fr else 1.0, "ascii_low_frac": float(np.mean([f < 0.8 for f in fr])) if fr else 0.0,
            "n_empty": int(sum(1 for t in texts if not t.strip()))}


def evaluate_gates(m: Dict[str, Any], gates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Automatic go / no-go = hard failures only (non-finite values). Tail mass, KL, truncation, coherence and entropy
    are reported as diagnostics. An optional ``tail_mass_max`` can be passed to re-enable a numerical gate explicitly."""
    g = {**DEFAULT_GATES, **(gates or {})}
    checks = {"finite": {"passed": bool(m["finite"]), "value": bool(m["finite"]), "rule": "no NaN/Inf in policy outputs, KL or training records"}}
    if "tail_mass_max" in g:
        checks["tail_mass"] = {"passed": bool(m["tail_mass"] < g["tail_mass_max"]), "value": m["tail_mass"], "rule": f"< {g['tail_mass_max']}"}
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks, "thresholds": g,
            "diagnostics_reported_not_gating": {k: m.get(k) for k in DIAGNOSTIC_ONLY}}


def collapse_check(m: Dict[str, Any], thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Continuation auto-stop: only a clear catastrophic failure (non-finite values); tail mass is reported, not gating,
    unless ``tail_mass_max`` is passed explicitly."""
    t = {**DEFAULT_COLLAPSE, **(thresholds or {})}
    reasons = []
    if not m.get("finite", True):
        reasons.append("non-finite values")
    if "tail_mass_max" in t and m["tail_mass"] > t["tail_mass_max"]:
        reasons.append(f"tail mass {m['tail_mass']:.3f} > {t['tail_mass_max']}")
    return {"collapsed": bool(reasons), "reasons": reasons, "thresholds": t,
            "diagnostics_reported_not_gating": {k: m.get(k) for k in DIAGNOSTIC_ONLY if k in m}}


def records_finite(records: Sequence[Dict[str, Any]], keys: Sequence[str] = ("loss", "kl_mean", "ratio_mean")) -> bool:
    for r in records:
        for k in keys:
            v = r.get(k)
            if isinstance(v, float) and not math.isfinite(v):
                return False
        for g in r.get("grad_norm", []):
            if not math.isfinite(g):
                return False
    return True
