"""DPO row construction: exactly two rows per underlying pair, correct orderings, no dedup."""
from conftest import make_pair, synthetic_pool
from saferl import data


def test_two_rows_with_both_objectives():
    rows = data.pair_to_dpo_rows(make_pair(0, better=1, safer=1))
    assert len(rows) == 2
    assert {r["objective"] for r in rows} == {"helpfulness", "harmlessness"}
    assert rows[0]["row_id"].endswith(":H") and rows[1]["row_id"].endswith(":S")
    assert rows[0]["pair_id"] == rows[1]["pair_id"]


def test_agreement_pair_same_ordering():
    p = make_pair(0, better=1, safer=1)
    h, s = data.pair_to_dpo_rows(p)
    assert h["chosen"] == s["chosen"] == p["response_1"]
    assert h["rejected"] == s["rejected"] == p["response_0"]
    assert h["chosen_id"] == s["chosen_id"] == 1


def test_conflict_pair_reversed_ordering():
    p = make_pair(0, better=0, safer=1)
    h, s = data.pair_to_dpo_rows(p)
    assert h["chosen"] == p["response_0"] and h["rejected"] == p["response_1"]
    assert s["chosen"] == p["response_1"] and s["rejected"] == p["response_0"]
    assert (h["chosen"], h["rejected"]) == (s["rejected"], s["chosen"])


def test_prompt_identical_and_objective_not_in_prompt():
    p = make_pair(0, better=0, safer=1)
    for r in data.pair_to_dpo_rows(p):
        assert r["prompt"] == p["prompt"]
        assert "helpfulness" not in r["prompt"] and "harmlessness" not in r["prompt"] and "objective" not in r["prompt"]


def test_no_deduplication_of_agreement_rows():
    pairs = synthetic_pool(500, "train", seed=9)
    rows = data.pairs_to_dpo_rows(pairs)
    assert len(rows) == 2 * len(pairs)
    assert sum(1 for r in rows if r["objective"] == "helpfulness") == len(pairs)
    assert sum(1 for r in rows if r["objective"] == "harmlessness") == len(pairs)
    agree = [p for p in pairs if p["pair_type"] == "agreement"]
    agree_rows = [r for r in rows if r["pair_type"] == "agreement"]
    assert len(agree_rows) == 2 * len(agree)  # identical (chosen, rejected) content kept twice
    assert len({r["row_id"] for r in rows}) == len(rows)
