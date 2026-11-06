"""Reward/cost-model rows derived from the SAME split manifests as the DPO arm.

Every underlying pair yields exactly two comparison rows with the same row ids as the DPO rows:
  <pair_id>:H  helpfulness  chosen = response[better_response_id], rejected = the other
  <pair_id>:S  harmlessness chosen = response[safer_response_id]  (= y_safe),  rejected = y_harmful
plus the pointwise PKU safety labels of both responses (used only by cost heads).

``objective`` is metadata: it never enters the model input (the text seen by the model is the
identical Qwen chat serialization of prompt + response for every row and architecture).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .data import OBJECTIVES
from .utils import read_jsonl

ARCHITECTURES = {
    # arch -> heads, objectives consumed, head used per objective, loss used per objective
    "scalar": {"heads": ["scalar"], "objectives": ["helpfulness", "harmlessness"],
               "head_for": {"helpfulness": "scalar", "harmlessness": "scalar"},
               "loss_for": {"helpfulness": "bt", "harmlessness": "bt"}},
    "shared": {"heads": ["reward", "cost"], "objectives": ["helpfulness", "harmlessness"],
               "head_for": {"helpfulness": "reward", "harmlessness": "cost"},
               "loss_for": {"helpfulness": "bt", "harmlessness": "cost"}},
    "separate_reward": {"heads": ["reward"], "objectives": ["helpfulness"],
                        "head_for": {"helpfulness": "reward"}, "loss_for": {"helpfulness": "bt"}},
    "separate_cost": {"heads": ["cost"], "objectives": ["harmlessness"],
                      "head_for": {"harmlessness": "cost"}, "loss_for": {"harmlessness": "cost"}},
}


def safety_sign(is_safe: bool) -> int:
    """s(y) = +1 if harmful/unsafe, -1 if safe (PKU ``is_response_X_safe`` field)."""
    return -1 if bool(is_safe) else 1


def pair_to_rm_rows(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    r = [pair["response_0"], pair["response_1"]]
    safe = [bool(pair["is_response_0_safe"]), bool(pair["is_response_1_safe"])]
    rows = []
    for objective, pref_id, suffix in (("helpfulness", pair["better_response_id"], "H"),
                                       ("harmlessness", pair["safer_response_id"], "S")):
        other = 1 - pref_id
        rows.append(OrderedDict(
            row_id=f"{pair['pair_id']}:{suffix}",
            pair_id=pair["pair_id"],
            objective=objective,
            pair_type=pair["pair_type"],
            prompt=pair["prompt"],
            chosen=r[pref_id],
            rejected=r[other],
            chosen_id=int(pref_id),
            rejected_id=int(other),
            chosen_is_safe=safe[pref_id],
            rejected_is_safe=safe[other],
            chosen_sign=safety_sign(safe[pref_id]),
            rejected_sign=safety_sign(safe[other]),
            better_response_id=pair["better_response_id"],
            safer_response_id=pair["safer_response_id"],
        ))
    return rows


def pairs_to_rm_rows(pairs: Iterable[Dict[str, Any]], objectives: Sequence[str] = OBJECTIVES) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in pairs:
        out.extend(r for r in pair_to_rm_rows(p) if r["objective"] in objectives)
    return out


def load_rm_rows(artifacts_dir: str, split: str, objectives: Sequence[str] = OBJECTIVES) -> List[Dict[str, Any]]:
    pairs = read_jsonl(Path(artifacts_dir) / "splits" / f"{split}.pairs.jsonl")
    return pairs_to_rm_rows(pairs, objectives)
