"""Subgroup DPO metrics (overall / objective / pair type / objective x pair type) and the
conflict-pair diagnostic. Pure numpy; inputs are DPO rows plus per-row policy/reference log-probs."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np

from .data import OBJECTIVES, PAIR_TYPES, normalize_prompt
from .utils import sha256_text

GROUP_ORDER = (
    "helpfulness/agreement", "helpfulness/conflict", "harmlessness/agreement", "harmlessness/conflict",
    "helpfulness", "harmlessness", "agreement", "conflict", "overall",
)


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def row_records(rows: Sequence[Dict[str, Any]], policy: Dict[str, Dict[str, float]], ref: Dict[str, Dict[str, float]],
                beta: float, label_smoothing: float = 0.0) -> List[Dict[str, Any]]:
    """Per-row loss / margin / rewards from summed log-probs (mirrors dpo_loss.dpo_loss)."""
    out = []
    for r in rows:
        p, q = policy[r["row_id"]], ref[r["row_id"]]
        chosen_reward = beta * (p["chosen"] - q["chosen"])
        rejected_reward = beta * (p["rejected"] - q["rejected"])
        margin = chosen_reward - rejected_reward
        loss = -_log_sigmoid(margin) * (1 - label_smoothing) - _log_sigmoid(-margin) * label_smoothing
        out.append({
            "row_id": r["row_id"], "pair_id": r["pair_id"], "objective": r["objective"], "pair_type": r["pair_type"],
            "prompt_hash": sha256_text(normalize_prompt(r["prompt"]))[:16] if "prompt" in r else r.get("prompt_hash", r["pair_id"]),
            "loss": float(loss), "margin": float(margin), "pref_correct": bool(margin > 0),
            "chosen_reward": chosen_reward, "rejected_reward": rejected_reward,
            "policy_chosen_logp": p["chosen"], "policy_rejected_logp": p["rejected"],
            "ref_chosen_logp": q["chosen"], "ref_rejected_logp": q["rejected"],
            "chosen_tokens": p.get("chosen_tokens"), "rejected_tokens": p.get("rejected_tokens"),
        })
    return out


STAT_KEYS = ("loss", "pref_acc", "pref_acc_tie_adjusted", "frac_margin_positive", "frac_margin_zero",
             "frac_margin_negative", "mean_margin", "chosen_reward", "rejected_reward",
             "policy_chosen_logp", "policy_rejected_logp", "ref_chosen_logp", "ref_rejected_logp")


def _row_values(r: Dict[str, Any]) -> Dict[str, float]:
    m = r["margin"]
    return {
        "loss": r["loss"],
        "pref_acc": 1.0 if m > 0 else 0.0,                       # strict: zero margin counts as incorrect
        "pref_acc_tie_adjusted": 1.0 if m > 0 else (0.5 if m == 0 else 0.0),  # zero margin contributes 0.5
        "frac_margin_positive": 1.0 if m > 0 else 0.0,
        "frac_margin_zero": 1.0 if m == 0 else 0.0,
        "frac_margin_negative": 1.0 if m < 0 else 0.0,
        "mean_margin": m,
        "chosen_reward": r["chosen_reward"], "rejected_reward": r["rejected_reward"],
        "policy_chosen_logp": r["policy_chosen_logp"], "policy_rejected_logp": r["policy_rejected_logp"],
        "ref_chosen_logp": r["ref_chosen_logp"], "ref_rejected_logp": r["ref_rejected_logp"],
    }


def _mean_of(values: List[Dict[str, float]]) -> Dict[str, float]:
    return {k: float(np.mean([v[k] for v in values])) for k in STAT_KEYS}


def _group_stats(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pair-averaged stats (every row weighs 1) plus prompt-averaged stats (every row's metric is
    first averaged within its prompt, then prompts are averaged with equal weight)."""
    if not recs:
        return {"n": 0, "n_prompts": 0}
    vals = [_row_values(r) for r in recs]
    out: Dict[str, Any] = {"n": len(recs)}
    out.update(_mean_of(vals))
    out["margin_std"] = float(np.std([r["margin"] for r in recs]))
    by_prompt: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for r, v in zip(recs, vals):
        by_prompt[r["prompt_hash"]].append(v)
    per_prompt = [_mean_of(v) for v in by_prompt.values()]
    out["n_prompts"] = len(per_prompt)
    out["max_rows_per_prompt"] = max(len(v) for v in by_prompt.values())
    out["prompt_averaged"] = _mean_of(per_prompt)
    return out


def summarize(records: Sequence[Dict[str, Any]], beta: float) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups["overall"].append(r)
        groups[r["objective"]].append(r)
        groups[r["pair_type"]].append(r)
        groups[f"{r['objective']}/{r['pair_type']}"].append(r)
    stats = {g: _group_stats(groups.get(g, [])) for g in GROUP_ORDER}
    h, s = stats["helpfulness"], stats["harmlessness"]
    combined = 0.5 * h["loss"] + 0.5 * s["loss"] if h["n"] and s["n"] else stats["overall"].get("loss")

    # Conflict diagnostic: for a conflict pair the H and S rows contain the same two responses with
    # opposite labels, so margin_S == -margin_H exactly. margin_H > 0 <=> policy favors the
    # helpfulness-preferred response; margin_H < 0 <=> favors the safety-preferred response.
    conflict_h = [r for r in records if r["pair_type"] == "conflict" and r["objective"] == "helpfulness"]
    n_c = len(conflict_h)
    fav_h = sum(1 for r in conflict_h if r["margin"] > 0)
    fav_s = sum(1 for r in conflict_h if r["margin"] < 0)
    ties = n_c - fav_h - fav_s
    by_prompt: Dict[str, List[float]] = defaultdict(list)
    for r in conflict_h:
        by_prompt[r["prompt_hash"]].append(r["margin"])
    pp = [np.mean(v) for v in by_prompt.values()]
    diag = {
        "n_conflict_pairs": n_c,
        "n_conflict_prompts": len(pp),
        "prompt_averaged": {
            "favors_helpfulness_preferred_frac": float(np.mean([np.mean([m > 0 for m in v]) for v in by_prompt.values()])) if pp else None,
            "favors_safety_preferred_frac": float(np.mean([np.mean([m < 0 for m in v]) for v in by_prompt.values()])) if pp else None,
            "mean_margin_toward_helpfulness_preferred": float(np.mean(pp)) if pp else None,
        },
        "favors_helpfulness_preferred": {"count": fav_h, "frac": (fav_h / n_c) if n_c else None},
        "favors_safety_preferred": {"count": fav_s, "frac": (fav_s / n_c) if n_c else None},
        "ties": {"count": ties, "frac": (ties / n_c) if n_c else None},
        "mean_abs_margin": float(np.mean([abs(r["margin"]) for r in conflict_h])) if n_c else None,
        "mean_margin_toward_helpfulness_preferred": float(np.mean([r["margin"] for r in conflict_h])) if n_c else None,
    }
    return {
        "beta": beta,
        "n_rows": len(records),
        "combined_loss": combined,
        "groups": stats,
        "conflict_diagnostic": diag,
    }


def _table(metrics: Dict[str, Any], prompt_averaged: bool) -> List[str]:
    lines = ["| group | n | n_prompts | loss | pref_acc | pref_acc_tie_adj | margin>0 | margin=0 | margin<0 | mean_margin | chosen_logp | rejected_logp |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in GROUP_ORDER:
        s = metrics["groups"][g]
        if s["n"] == 0:
            lines.append(f"| {g} | 0 | 0 | - | - | - | - | - | - | - | - | - |")
            continue
        v = s["prompt_averaged"] if prompt_averaged else s
        lines.append(f"| {g} | {s['n']} | {s['n_prompts']} | {v['loss']:.4f} | {v['pref_acc']:.4f} | {v['pref_acc_tie_adjusted']:.4f} | "
                     f"{v['frac_margin_positive']:.3f} | {v['frac_margin_zero']:.3f} | {v['frac_margin_negative']:.3f} | {v['mean_margin']:.4f} | "
                     f"{v['policy_chosen_logp']:.2f} | {v['policy_rejected_logp']:.2f} |")
    return lines


def render_metrics_markdown(metrics: Dict[str, Any], title: str) -> str:
    lines = [f"### {title}", "", "**Pair-averaged** (each DPO row weighs 1)", ""]
    lines += _table(metrics, prompt_averaged=False)
    lines += ["", "**Prompt-averaged** (rows averaged within prompt, then prompts weighed equally)", ""]
    lines += _table(metrics, prompt_averaged=True)
    lines.append("")
    lines.append(f"combined loss (0.5 H + 0.5 S, pair-averaged; used for checkpoint selection): {metrics['combined_loss']:.4f}")
    d = metrics["conflict_diagnostic"]
    if d["n_conflict_pairs"]:
        pa = d["prompt_averaged"]
        lines.append(f"conflict pairs: n={d['n_conflict_pairs']} ({d['n_conflict_prompts']} prompts); policy favors helpfulness-preferred response in "
                     f"{d['favors_helpfulness_preferred']['count']} ({100*d['favors_helpfulness_preferred']['frac']:.1f}%), "
                     f"safety-preferred in {d['favors_safety_preferred']['count']} ({100*d['favors_safety_preferred']['frac']:.1f}%), "
                     f"ties {d['ties']['count']} ({100*d['ties']['frac']:.1f}%); mean margin toward helpfulness-preferred = {d['mean_margin_toward_helpfulness_preferred']:.4f}; "
                     f"prompt-averaged: helpfulness {100*pa['favors_helpfulness_preferred_frac']:.1f}%, safety {100*pa['favors_safety_preferred_frac']:.1f}%, "
                     f"margin {pa['mean_margin_toward_helpfulness_preferred']:.4f}")
    lines.append("")
    return "\n".join(lines)


def is_finite_metrics(metrics: Dict[str, Any]) -> bool:
    v = metrics.get("combined_loss")
    return v is not None and math.isfinite(v)
