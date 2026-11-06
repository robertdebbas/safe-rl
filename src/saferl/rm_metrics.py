"""Reward/cost model metrics: per objective (helpfulness / harmlessness) x pair type, pair-averaged
and prompt-averaged, with the same tie conventions as the DPO metrics.

Row record fields (from rm_rows + scores):
  objective, pair_type, prompt_hash, score_chosen, score_rejected, margin (positive = ranking
  correct: reward/scalar chosen - rejected; cost harmful - safe), loss (objective loss used in
  training: BT for helpfulness/scalar rows, rank + class for cost rows) and, for cost rows,
  rank_loss, class_loss, the pointwise labels and classification correctness at threshold C = 0.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np

from .data import normalize_prompt
from .utils import sha256_text

GROUP_ORDER = (
    "helpfulness/agreement", "helpfulness/conflict", "helpfulness",
    "harmlessness/agreement", "harmlessness/conflict", "harmlessness", "overall",
)
RANK_KEYS = ("loss", "rank_loss", "acc", "acc_tie_adjusted", "frac_margin_positive", "frac_margin_zero",
             "frac_margin_negative", "mean_margin", "mean_score_chosen", "mean_score_rejected")
COST_KEYS = ("mean_cost_safe", "mean_cost_unsafe", "cost_gap_unsafe_minus_safe")  # audit-only descriptives (shift-invariant gap)
CLASS_KEYS = ("class_loss", "class_acc", "class_acc_safe", "class_acc_unsafe")        # only with the pointwise ablation


def _log_sigmoid(x: float) -> float:
    return float(-np.logaddexp(0.0, -x))


def row_records(rows: Sequence[Dict[str, Any]], scores: Dict[str, Dict[str, float]], loss_for: Dict[str, str],
                cost_pointwise_classification: bool = False) -> List[Dict[str, Any]]:
    """``scores[row_id] = {"chosen": float, "rejected": float}`` under the head used for that objective.
    Cost rows: loss = pairwise BT on (harmful - safer); the pointwise terms are added only in the ablation."""
    out = []
    for r in rows:
        s = scores[r["row_id"]]
        rec: Dict[str, Any] = {
            "row_id": r["row_id"], "pair_id": r["pair_id"], "objective": r["objective"], "pair_type": r["pair_type"],
            "prompt_hash": sha256_text(normalize_prompt(r["prompt"]))[:16],
            "score_chosen": s["chosen"], "score_rejected": s["rejected"],
        }
        if loss_for[r["objective"]] == "bt":
            m = s["chosen"] - s["rejected"]
            rec.update({"margin": m, "rank_loss": -_log_sigmoid(m), "loss": -_log_sigmoid(m), "is_cost": False})
        else:  # cost: chosen = y_safer (safer_response_id), rejected = y_harmful
            m = s["rejected"] - s["chosen"]
            rank = -_log_sigmoid(m)
            cls = (-_log_sigmoid(r["chosen_sign"] * s["chosen"]) - _log_sigmoid(r["rejected_sign"] * s["rejected"])) \
                if cost_pointwise_classification else 0.0
            rec.update({
                "margin": m, "rank_loss": rank, "class_loss": cls, "loss": rank + cls, "is_cost": True,
                "classification_in_loss": bool(cost_pointwise_classification),
                "responses": [
                    {"cost": s["chosen"], "unsafe": not r["chosen_is_safe"], "correct": (s["chosen"] > 0) == (not r["chosen_is_safe"])},
                    {"cost": s["rejected"], "unsafe": not r["rejected_is_safe"], "correct": (s["rejected"] > 0) == (not r["rejected_is_safe"])},
                ],
            })
        out.append(rec)
    return out


def _row_values(r: Dict[str, Any]) -> Dict[str, float]:
    m = r["margin"]
    v = {
        "loss": r["loss"], "rank_loss": r["rank_loss"],
        "acc": 1.0 if m > 0 else 0.0,
        "acc_tie_adjusted": 1.0 if m > 0 else (0.5 if m == 0 else 0.0),
        "frac_margin_positive": 1.0 if m > 0 else 0.0, "frac_margin_zero": 1.0 if m == 0 else 0.0,
        "frac_margin_negative": 1.0 if m < 0 else 0.0,
        "mean_margin": m, "mean_score_chosen": r["score_chosen"], "mean_score_rejected": r["score_rejected"],
    }
    if r.get("is_cost"):
        resp = r["responses"]
        safe = [x for x in resp if not x["unsafe"]]
        unsafe = [x for x in resp if x["unsafe"]]
        ms = float(np.mean([x["cost"] for x in safe])) if safe else float("nan")
        mu = float(np.mean([x["cost"] for x in unsafe])) if unsafe else float("nan")
        v.update({"mean_cost_safe": ms, "mean_cost_unsafe": mu,
                  "cost_gap_unsafe_minus_safe": (mu - ms) if (safe and unsafe) else float("nan")})
        if r.get("classification_in_loss"):
            v.update({
                "class_loss": r["class_loss"],
                "class_acc": float(np.mean([x["correct"] for x in resp])),
                "class_acc_safe": float(np.mean([x["correct"] for x in safe])) if safe else float("nan"),
                "class_acc_unsafe": float(np.mean([x["correct"] for x in unsafe])) if unsafe else float("nan"),
            })
    return v


def _nanmean(values: List[Dict[str, float]], keys: Sequence[str]) -> Dict[str, float]:
    out = {}
    for k in keys:
        arr = np.array([v[k] for v in values if k in v], dtype=float)
        out[k] = float(np.nanmean(arr)) if arr.size and not np.all(np.isnan(arr)) else None
    return out


def _group_stats(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not recs:
        return {"n": 0, "n_prompts": 0}
    has_cost = any(r.get("is_cost") for r in recs)
    has_class = any(r.get("classification_in_loss") for r in recs)
    keys = RANK_KEYS + (COST_KEYS if has_cost else ()) + (CLASS_KEYS if has_class else ())
    vals = [_row_values(r) for r in recs]
    out: Dict[str, Any] = {"n": len(recs)}
    out.update(_nanmean(vals, keys))
    if has_cost:  # pooled pointwise counts (every response weighs 1)
        resp = [x for r in recs if r.get("is_cost") for x in r["responses"]]
        out["n_responses"] = len(resp)
        out["n_responses_safe"] = sum(1 for x in resp if not x["unsafe"])
        out["n_responses_unsafe"] = sum(1 for x in resp if x["unsafe"])
        if has_class:
            out["class_acc_pooled"] = float(np.mean([x["correct"] for x in resp]))
    by_prompt: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for r, v in zip(recs, vals):
        by_prompt[r["prompt_hash"]].append(v)
    per_prompt = [_nanmean(v, keys) for v in by_prompt.values()]
    out["n_prompts"] = len(per_prompt)
    out["max_rows_per_prompt"] = max(len(v) for v in by_prompt.values())
    out["prompt_averaged"] = _nanmean([{k: (np.nan if x is None else x) for k, x in p.items()} for p in per_prompt], keys)
    return out


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups["overall"].append(r)
        groups[r["objective"]].append(r)
        groups[f"{r['objective']}/{r['pair_type']}"].append(r)
    stats = {g: _group_stats(groups.get(g, [])) for g in GROUP_ORDER}
    h, s = stats["helpfulness"], stats["harmlessness"]
    if h["n"] and s["n"]:
        combined = 0.5 * h["loss"] + 0.5 * s["loss"]
    else:
        combined = h["loss"] if h["n"] else s["loss"]
    return {"n_rows": len(records), "combined_loss": combined, "groups": stats}


def _fmt(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def _table(metrics: Dict[str, Any], prompt_averaged: bool) -> List[str]:
    cols = ["group", "n", "n_prompts", "loss", "rank_loss", "acc", "acc_tie_adj", "margin>0", "margin=0", "margin<0",
            "mean_margin", "cost_safe", "cost_unsafe", "cost_gap", "class_loss", "class_acc"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for g in GROUP_ORDER:
        s = metrics["groups"][g]
        if s["n"] == 0:
            continue
        v = s["prompt_averaged"] if prompt_averaged else s
        lines.append(f"| {g} | {s['n']} | {s['n_prompts']} | {_fmt(v['loss'])} | {_fmt(v['rank_loss'])} | {_fmt(v['acc'])} | "
                     f"{_fmt(v['acc_tie_adjusted'])} | {_fmt(v['frac_margin_positive'])} | {_fmt(v['frac_margin_zero'])} | "
                     f"{_fmt(v['frac_margin_negative'])} | {_fmt(v['mean_margin'])} | {_fmt(v.get('mean_cost_safe'))} | "
                     f"{_fmt(v.get('mean_cost_unsafe'))} | {_fmt(v.get('cost_gap_unsafe_minus_safe'))} | {_fmt(v.get('class_loss'))} | {_fmt(v.get('class_acc'))} |")
    return lines


def render_metrics_markdown(metrics: Dict[str, Any], title: str) -> str:
    lines = [f"### {title}", "", "**Pair-averaged** (each comparison row weighs 1; cost_safe/cost_unsafe/cost_gap are audit-only descriptives "
             "from PKU pointwise labels, not used in the primary loss; class_* appear only in the pointwise ablation)", ""]
    lines += _table(metrics, False)
    lines += ["", "**Prompt-averaged** (rows averaged within prompt, then prompts weighed equally)", ""]
    lines += _table(metrics, True)
    lines += ["", f"combined loss (0.5 H + 0.5 S when both present; used for checkpoint selection): {metrics['combined_loss']:.4f}", ""]
    return "\n".join(lines)
