"""Metric semantics: margin-sign fractions, tie-adjusted preference accuracy, and prompt averaging."""
import math

import pytest

from saferl.metrics import row_records, summarize


def _rows_and_logps():
    """Three prompts: P1 has 3 pairs, P2 and P3 one pair each. Margins (helpfulness rows):
    P1: +1, +1, +1 ; P2: 0 ; P3: -1.   Harmlessness rows mirror them (agreement pairs)."""
    spec = [("P1", "a", 1.0), ("P1", "b", 1.0), ("P1", "c", 1.0), ("P2", "d", 0.0), ("P3", "e", -1.0)]
    rows, policy, ref = [], {}, {}
    for prompt, pid, margin in spec:
        for obj, suffix in (("helpfulness", "H"), ("harmlessness", "S")):
            rid = f"{pid}:{suffix}"
            rows.append({"row_id": rid, "pair_id": pid, "objective": obj, "pair_type": "agreement",
                         "prompt": f"  {prompt}  ", "chosen": "x", "rejected": "y"})
            # beta = 0.1 -> margin = 0.1 * ((pc - rc) - (pr - rr)); set pc - rc = 10*margin, pr == rr
            policy[rid] = {"chosen": -10.0 + 10 * margin, "rejected": -10.0, "chosen_tokens": 1, "rejected_tokens": 1}
            ref[rid] = {"chosen": -10.0, "rejected": -10.0}
    return rows, policy, ref


def test_margin_fractions_and_tie_adjusted_accuracy():
    rows, policy, ref = _rows_and_logps()
    m = summarize(row_records(rows, policy, ref, beta=0.1), beta=0.1)
    g = m["groups"]["overall"]
    assert g["n"] == 10 and g["n_prompts"] == 3 and g["max_rows_per_prompt"] == 6
    # pair-averaged: 6 positive, 2 zero, 2 negative rows
    assert math.isclose(g["frac_margin_positive"], 0.6) and math.isclose(g["frac_margin_zero"], 0.2) and math.isclose(g["frac_margin_negative"], 0.2)
    assert math.isclose(g["pref_acc"], 0.6)                      # strict: ties incorrect
    assert math.isclose(g["pref_acc_tie_adjusted"], 0.6 + 0.5 * 0.2)  # ties contribute 0.5
    assert math.isclose(g["frac_margin_positive"] + g["frac_margin_zero"] + g["frac_margin_negative"], 1.0)


def test_prompt_averaging_weights_prompts_equally():
    rows, policy, ref = _rows_and_logps()
    m = summarize(row_records(rows, policy, ref, beta=0.1), beta=0.1)
    pa = m["groups"]["overall"]["prompt_averaged"]
    # per prompt: P1 -> acc 1, P2 -> acc 0 (tie-adj 0.5), P3 -> acc 0 ; equal weights over 3 prompts
    assert math.isclose(pa["pref_acc"], 1 / 3)
    assert math.isclose(pa["pref_acc_tie_adjusted"], (1 + 0.5 + 0) / 3)
    assert math.isclose(pa["frac_margin_zero"], 1 / 3) and math.isclose(pa["frac_margin_negative"], 1 / 3)
    assert math.isclose(pa["mean_margin"], (1 + 0 - 1) / 3)
    # pair-averaged mean margin is dominated by P1's three pairs
    assert math.isclose(m["groups"]["overall"]["mean_margin"], (3 - 1) / 5)
    # prompt normalization: "  P1  " and "P1" are the same prompt
    rows2 = [dict(r, prompt=r["prompt"].strip()) if r["pair_id"] == "a" else r for r in rows]
    m2 = summarize(row_records(rows2, policy, ref, beta=0.1), beta=0.1)
    assert m2["groups"]["overall"]["n_prompts"] == 3


def test_conflict_diagnostic_counts_ties_and_prompt_average():
    rows, policy, ref = _rows_and_logps()
    for r in rows:
        r["pair_type"] = "conflict"
    m = summarize(row_records(rows, policy, ref, beta=0.1), beta=0.1)
    d = m["conflict_diagnostic"]
    assert d["n_conflict_pairs"] == 5 and d["n_conflict_prompts"] == 3
    assert d["favors_helpfulness_preferred"]["count"] == 3 and d["favors_safety_preferred"]["count"] == 1 and d["ties"]["count"] == 1
    assert math.isclose(d["prompt_averaged"]["favors_helpfulness_preferred_frac"], 1 / 3)
    assert math.isclose(d["prompt_averaged"]["favors_safety_preferred_frac"], 1 / 3)
