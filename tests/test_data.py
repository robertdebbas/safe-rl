"""Data pipeline unit tests on synthetic pools (fast, no downloads)."""
from collections import Counter

import numpy as np
import pytest

from conftest import make_pair, synthetic_pool
from saferl import data


def test_validity_filtering_counts_each_reason():
    pairs = [
        make_pair(0),
        make_pair(1, prompt="   "),
        make_pair(2, r0="  "),
        make_pair(3, r1=""),
        make_pair(4, better=2),
        make_pair(5, safer=None),
        make_pair(6, r0="", better=7),
    ]
    kept, counts = data.filter_valid(pairs)
    assert [p["official_index"] for p in kept] == [0]
    assert counts["empty_prompt"] == 1
    assert counts["empty_response_0"] == 2
    assert counts["empty_response_1"] == 1
    assert counts["invalid_better_response_id"] == 2
    assert counts["invalid_safer_response_id"] == 1
    assert counts["removed_total"] == 6
    assert counts["removed_with_multiple_reasons"] == 1
    assert counts["kept"] == 1


def test_pair_type_and_normalization():
    p = make_pair(0, prompt="  hello   world \n", better=0, safer=1)
    assert p["pair_type"] == "conflict"
    assert p["prompt_key"] == "hello world"
    assert p["prompt"] == "  hello   world \n"  # original text untouched
    assert make_pair(1, better=1, safer=1)["pair_type"] == "agreement"
    assert make_pair(2, better=3, safer=1)["pair_type"] is None


def test_leakage_removal_by_normalized_prompt():
    train = [make_pair(0, prompt="A  b"), make_pair(1, prompt="c"), make_pair(2, prompt="a b")]
    kept, rep = data.remove_prompt_leakage(train, {"A b", "zzz"})
    assert [p["official_index"] for p in kept] == [1, 2]
    assert rep["removed_pairs"] == 1 and rep["removed_unique_prompts"] == 1


def test_largest_remainder_sums_exactly():
    alloc = data.largest_remainder({"a": 10, "b": 10, "c": 1}, 7)
    assert sum(alloc.values()) == 7
    assert alloc["a"] == alloc["b"] == 3 and alloc["c"] == 1
    assert sum(data.largest_remainder({"x": 0.3, "y": 0.3, "z": 0.4}, 2000).values()) == 2000


def _splits(seed=42, n_train=4000, n_test=1200):
    train_pool = synthetic_pool(n_train, "train", seed=1, conflict_frac=0.3)
    test_pool = synthetic_pool(n_test, "test", seed=2, conflict_frac=0.25, start=100000)
    sizes = {"train": {"agreement": 300, "conflict": 300}, "val": {"agreement": 60, "conflict": 60},
             "test_balanced": {"agreement": 100, "conflict": 100}, "test_representative": {"total": 200}}
    return data.build_splits(train_pool, test_pool, sizes, data_seed=seed), train_pool, test_pool


def test_sampling_is_deterministic_for_fixed_seed():
    (s1, _), _, _ = _splits(42)
    (s2, _), _, _ = _splits(42)
    for name in s1:
        assert [p["pair_id"] for p in s1[name]] == [p["pair_id"] for p in s2[name]]
    (s3, _), _, _ = _splits(7)
    assert [p["pair_id"] for p in s3["train"]] != [p["pair_id"] for p in s1["train"]]


def test_sampling_independent_of_input_order():
    train_pool = synthetic_pool(3000, "train", seed=1)
    test_pool = synthetic_pool(800, "test", seed=2, start=100000)
    sizes = {"train": {"agreement": 200, "conflict": 200}, "val": {"agreement": 40, "conflict": 40},
             "test_balanced": {"agreement": 50, "conflict": 50}, "test_representative": {"total": 100}}
    a, _ = data.build_splits(train_pool, test_pool, sizes, 42)
    b, _ = data.build_splits(train_pool[::-1], test_pool[::-1], sizes, 42)
    for n in a:
        assert [p["pair_id"] for p in a[n]] == [p["pair_id"] for p in b[n]]


def test_exact_quotas_and_prompt_disjointness():
    (splits, rep), train_pool, test_pool = _splits()
    st = {n: data.agreement_stats(s) for n, s in splits.items()}
    assert (st["train"]["agreement"], st["train"]["conflict"]) == (300, 300)
    assert (st["val"]["agreement"], st["val"]["conflict"]) == (60, 60)
    assert (st["test_balanced"]["agreement"], st["test_balanced"]["conflict"]) == (100, 100)
    assert st["test_representative"]["n"] == 200
    overlap = data.check_prompt_disjoint(splits)
    assert all(v == 0 for v in overlap.values()), overlap
    # official test never in train/val; test subsets only from official test
    assert all(p["official_split"] == "train" for n in ("train", "val") for p in splits[n])
    assert all(p["official_split"] == "test" for n in ("test_balanced", "test_representative") for p in splits[n])
    # no duplicates anywhere
    ids = [p["pair_id"] for s in splits.values() for p in s]
    assert len(ids) == len(set(ids))


def test_representative_matches_natural_distribution():
    (splits, rep), train_pool, test_pool = _splits()
    natural = data.agreement_stats(test_pool)["conflict_pct"] / 100
    got = data.agreement_stats(splits["test_representative"])["conflict"] / 200
    assert abs(got - natural) <= 0.5 / 200 + 1e-9  # exact largest-remainder rounding
    assert rep["test_representative"]["quotas"] == data.natural_quotas(test_pool, 200)


def test_stratified_sampling_tracks_reference_proportions():
    pool = synthetic_pool(5000, "train", seed=3)
    conflict = [p for p in pool if p["pair_type"] == "conflict"]
    rng = np.random.default_rng(0)
    sel, rep = data.stratified_sample(conflict, 400, rng)
    assert len(sel) == 400 and len({p["pair_id"] for p in sel}) == 400
    ref = Counter(p["prompt_source"] for p in conflict)
    got = Counter(p["prompt_source"] for p in sel)
    for k in ref:
        assert abs(got[k] / 400 - ref[k] / len(conflict)) < 0.03


def test_stratified_sampling_reports_deficit_and_fails_when_impossible():
    pool = synthetic_pool(50, "train", seed=4)
    rng = np.random.default_rng(0)
    sel, rep = data.stratified_sample(pool, 50, rng)
    assert len(sel) == 50
    with pytest.raises(ValueError):
        data.stratified_sample(pool, 51, np.random.default_rng(0))


def test_max_pairs_per_prompt_cap():
    pool = synthetic_pool(2000, "train", seed=5, n_prompts=100)
    rng = np.random.default_rng(0)
    sel, rep = data.stratified_sample(pool, 150, rng, max_pairs_per_prompt=2)
    assert len(sel) == 150
    assert max(Counter(p["prompt_key"] for p in sel).values()) <= 2
