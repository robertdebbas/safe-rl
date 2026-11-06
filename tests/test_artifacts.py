"""Invariants of the generated artifacts (run scripts/prepare_data.py first) and byte-identical
regeneration of the split manifests."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from saferl.data import normalize_prompt
from saferl.utils import read_json, read_jsonl, sha256_file

pytestmark = pytest.mark.artifacts

SPLITS = ("train", "val", "test_balanced", "test_representative")


def _manifests(artifacts_dir):
    return {s: read_jsonl(artifacts_dir / "splits" / f"{s}.manifest.jsonl") for s in SPLITS}


def test_exact_counts(artifacts_dir):
    m = _manifests(artifacts_dir)
    c = {s: Counter(r["pair_type"] for r in m[s]) for s in SPLITS}
    assert c["train"] == Counter(agreement=5000, conflict=5000)
    assert c["val"] == Counter(agreement=1000, conflict=1000)
    assert c["test_balanced"] == Counter(agreement=1000, conflict=1000)
    assert sum(c["test_representative"].values()) == 2000
    audit = read_json(artifacts_dir / "data_audit.json")
    natural = audit["official_agreement"]["test"]["conflict_pct"] / 100
    assert abs(c["test_representative"]["conflict"] / 2000 - natural) < 0.01


def test_official_test_never_in_train_or_val(artifacts_dir):
    m = _manifests(artifacts_dir)
    assert all(r["official_split"] == "train" for s in ("train", "val") for r in m[s])
    assert all(r["official_split"] == "test" for s in ("test_balanced", "test_representative") for r in m[s])
    # no official-train pair reused as test and vice versa (by official index)
    tr = {(r["official_split"], r["official_index"]) for s in ("train", "val") for r in m[s]}
    te = {(r["official_split"], r["official_index"]) for s in ("test_balanced", "test_representative") for r in m[s]}
    assert not (tr & te)


def test_no_normalized_prompt_leakage(artifacts_dir):
    pairs = {s: read_jsonl(artifacts_dir / "splits" / f"{s}.pairs.jsonl") for s in SPLITS}
    keys = {s: {normalize_prompt(p["prompt"]) for p in pairs[s]} for s in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not (keys[a] & keys[b]), (a, b)
    # train/val prompts must not appear anywhere in the official test split either
    audit = read_json(artifacts_dir / "data_audit.json")
    assert audit["official_test_in_train_val"] == 0
    assert all(v == 0 for v in audit["prompt_disjointness"].values())


def test_dpo_rows_match_pairs(artifacts_dir):
    for s in SPLITS:
        pairs = read_jsonl(artifacts_dir / "splits" / f"{s}.pairs.jsonl")
        rows = read_jsonl(artifacts_dir / "dpo" / f"{s}.jsonl")
        assert len(rows) == 2 * len(pairs)
        by_pair = {}
        for r in rows:
            by_pair.setdefault(r["pair_id"], []).append(r)
        for p in pairs:
            h, sx = sorted(by_pair[p["pair_id"]], key=lambda r: r["objective"] != "helpfulness")
            assert h["objective"] == "helpfulness" and sx["objective"] == "harmlessness"
            assert h["prompt"] == sx["prompt"] == p["prompt"]
            resp = [p["response_0"], p["response_1"]]
            assert h["chosen"] == resp[p["better_response_id"]] and sx["chosen"] == resp[p["safer_response_id"]]
            if p["pair_type"] == "conflict":
                assert (h["chosen"], h["rejected"]) == (sx["rejected"], sx["chosen"])
            else:
                assert (h["chosen"], h["rejected"]) == (sx["chosen"], sx["rejected"])


def test_recorded_hashes_match_files(artifacts_dir):
    h = read_json(artifacts_dir / "splits" / "manifest_hashes.json")
    for s in SPLITS:
        assert h[s]["manifest_sha256"] == sha256_file(artifacts_dir / "splits" / f"{s}.manifest.jsonl")
        assert h[s]["dpo_rows_sha256"] == sha256_file(artifacts_dir / "dpo" / f"{s}.jsonl")


def test_empty_rows_excluded(artifacts_dir):
    for s in SPLITS:
        for p in read_jsonl(artifacts_dir / "splits" / f"{s}.pairs.jsonl"):
            assert p["prompt"].strip() and p["response_0"].strip() and p["response_1"].strip()
            assert p["better_response_id"] in (0, 1) and p["safer_response_id"] in (0, 1)


@pytest.mark.slow
def test_regeneration_is_byte_identical(smoke_artifacts_dir, tmp_path):
    """Re-run the smoke data config into a temp dir; manifests must be byte-identical."""
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-W", "ignore", str(root / "scripts" / "prepare_data.py"), "--config",
                    str(root / "configs" / "data_smoke.yaml"), "--set", f"out_dir={tmp_path}"], check=True,
                   cwd=root, capture_output=True)
    for s in SPLITS:
        assert sha256_file(tmp_path / "splits" / f"{s}.manifest.jsonl") == sha256_file(smoke_artifacts_dir / "splits" / f"{s}.manifest.jsonl")
        assert sha256_file(tmp_path / "dpo" / f"{s}.jsonl") == sha256_file(smoke_artifacts_dir / "dpo" / f"{s}.jsonl")
    assert read_json(tmp_path / "splits" / "manifest_hashes.json") == read_json(smoke_artifacts_dir / "splits" / "manifest_hashes.json")
