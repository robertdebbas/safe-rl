"""End-to-end data preparation: official dataset -> audited, prompt-disjoint, fixed splits.

Run via ``scripts/prepare_data.py --config configs/data.yaml``. Everything written under
``cfg['out_dir']`` is fully determined by (dataset revision, tokenizer, config, data_seed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from . import audit, data
from .formatting import assert_no_objective_leakage, serialize_for_inspection
from .utils import get_logger, read_json, sha256_file, write_json, write_jsonl

log = get_logger("saferl.prepare")

SPLIT_NAMES = ("train", "val", "test_balanced", "test_representative")


class LengthExclusionTooLarge(RuntimeError):
    pass


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def run_prepare(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data_seed = int(cfg["data_seed"])
    report: Dict[str, Any] = {"data_seed": data_seed, "tokenizer": cfg["tokenizer"], "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}

    # 1. official dataset -------------------------------------------------------------------
    ds, meta = data.load_official(cfg["revision"], cfg.get("config", "default"), cfg.get("dataset", data.DATASET_NAME))
    report["source"] = meta
    write_json(out_dir / "schema.json", meta["schema"])
    log.info("official sizes %s", meta["official_split_sizes"])

    raw = {s: data.dataset_to_pairs(ds[s], s) for s in ("train", "test")}
    report["official_agreement"] = {s: data.agreement_stats(raw[s]) for s in raw}
    report["official_agreement"]["train+test"] = data.agreement_stats(raw["train"] + raw["test"])
    for s, st in report["official_agreement"].items():
        log.info("official %s: n=%d agreement=%d (%.4f%%) conflict=%d (%.4f%%)", s, st["n"], st["agreement"],
                 st["agreement_pct"] or 0, st["conflict"], st["conflict_pct"] or 0)

    # 2. validity ---------------------------------------------------------------------------
    valid, report["validity"] = {}, {}
    for s in raw:
        valid[s], report["validity"][s] = data.filter_valid(raw[s])
        log.info("validity %s: %s", s, report["validity"][s])

    # 3. leakage: official test prompts (ALL rows, even invalid ones) are forbidden in train ---
    test_prompt_keys = {p["prompt_key"] for p in raw["test"]}
    train_pool, lk = data.remove_prompt_leakage(valid["train"], test_prompt_keys)
    train_keys = {p["prompt_key"] for p in raw["train"]}
    lk.update({
        "unique_train_prompts": len(train_keys),
        "unique_test_prompts": len(test_prompt_keys),
        "shared_normalized_prompts": len(train_keys & test_prompt_keys),
    })
    report["leakage"] = lk
    test_pool = valid["test"]
    log.info("leakage: %s", lk)

    # 4. token-length audit -----------------------------------------------------------------
    tok = _load_tokenizer(cfg["tokenizer"])
    lengths = audit.compute_lengths(tok, train_pool + test_pool)
    train_ids = [p["pair_id"] for p in train_pool]
    test_ids = [p["pair_id"] for p in test_pool]
    all_ids = train_ids + test_ids
    report["lengths"] = {
        "quantiles": {
            "eligible_train_pool": audit.length_quantiles(lengths, train_ids),
            "eligible_test_pool": audit.length_quantiles(lengths, test_ids),
            "combined_eligible_pool": audit.length_quantiles(lengths, all_ids),
        }
    }
    prompt_vals = [lengths[i]["prompt"] for i in all_ids]
    total_vals = [lengths[i]["total_0"] for i in all_ids] + [lengths[i]["total_1"] for i in all_ids]
    pcands, tcands = cfg["prompt_length_candidates"], cfg["total_length_candidates"]
    target = float(cfg["coverage_target"])
    report["lengths"]["coverage"] = {"prompt": audit.coverage(prompt_vals, pcands), "total": audit.coverage(total_vals, tcands)}
    rec_prompt = audit.choose_threshold(prompt_vals, pcands, target)
    rec_total = audit.choose_threshold(total_vals, tcands, target)
    override = cfg.get("length_limits") or {}
    if override.get("max_prompt_length") and override.get("max_length"):
        max_prompt_length, max_length, source = int(override["max_prompt_length"]), int(override["max_length"]), "config override"
    else:
        if rec_prompt is None or rec_total is None:
            raise LengthExclusionTooLarge(
                f"No candidate threshold reaches coverage {target}: prompt->{rec_prompt}, total->{rec_total}. "
                f"Coverage: {report['lengths']['coverage']}")
        max_prompt_length, max_length, source = rec_prompt, rec_total, "audit (smallest candidate with coverage >= target)"
    p_cov = sum(1 for v in prompt_vals if v <= max_prompt_length) / len(prompt_vals)
    t_cov = sum(1 for v in total_vals if v <= max_length) / len(total_vals)
    report["length_limits"] = {
        "max_prompt_length": max_prompt_length, "max_length": max_length, "source": source,
        "recommended_from_audit": {"max_prompt_length": rec_prompt, "max_length": rec_total},
        "coverage_target": target, "prompt_coverage": p_cov, "total_coverage": t_cov,
        "coverage_target_met": bool(p_cov >= target and t_cov >= target),
    }
    log.info("length limits: %s", report["length_limits"])

    # 5. exclude over-length underlying pairs BEFORE sampling ---------------------------------
    train_pool, ex_tr = data.apply_length_limits(train_pool, lengths, max_prompt_length, max_length)
    test_pool, ex_te = data.apply_length_limits(test_pool, lengths, max_prompt_length, max_length)
    frac = (ex_tr["removed_total"] + ex_te["removed_total"]) / max(1, ex_tr["input"] + ex_te["input"])
    report["length_exclusion"] = {"train": ex_tr, "test": ex_te, "fraction_excluded_combined": frac}
    log.info("length exclusion: %s", report["length_exclusion"])
    if frac > float(cfg["max_exclusion_fraction"]) and not cfg.get("allow_large_exclusion", False):
        write_json(out_dir / "data_audit.json", report)
        raise LengthExclusionTooLarge(
            f"{100*frac:.3f}% of eligible pairs exceed the limits (max allowed {100*float(cfg['max_exclusion_fraction']):.2f}%). "
            "Stopping before sampling; inspect artifacts/data_audit.json and decide.")

    report["pools"] = {"train": audit.pool_summary(train_pool), "test": audit.pool_summary(test_pool)}

    # 6. sampling -----------------------------------------------------------------------------
    splits, samp = data.build_splits(train_pool, test_pool, cfg["splits"], data_seed,
                                     max_pairs_per_prompt=cfg.get("max_pairs_per_prompt"))
    report["sampling_report"] = samp

    # 7. checks -------------------------------------------------------------------------------
    report["prompt_disjointness"] = data.check_prompt_disjoint(splits)
    assert all(v == 0 for v in report["prompt_disjointness"].values()), report["prompt_disjointness"]
    test_in_trainval = sum(1 for n in ("train", "val") for p in splits[n] if p["official_split"] == "test")
    test_in_trainval += sum(1 for n in ("train", "val") for p in splits[n] if p["prompt_key"] in test_prompt_keys)
    report["official_test_in_train_val"] = test_in_trainval
    assert test_in_trainval == 0
    for n in ("test_balanced", "test_representative"):
        assert all(p["official_split"] == "test" for p in splits[n])
    for n, q in cfg["splits"].items():
        got = data.agreement_stats(splits[n])
        for t in data.PAIR_TYPES:
            if t in q:
                assert got[t] == q[t], (n, t, got[t], q[t])
        if "total" in q:
            assert got["n"] == q["total"], (n, got["n"], q["total"])

    # 8. write artifacts ----------------------------------------------------------------------
    split_dir, dpo_dir = out_dir / "splits", out_dir / "dpo"
    report["splits"], hashes = {}, {}
    for n in SPLIT_NAMES:
        pairs = splits[n]
        rows = data.pairs_to_dpo_rows(pairs)
        write_jsonl(split_dir / f"{n}.manifest.jsonl", (data.manifest_record(p) for p in pairs))
        write_jsonl(split_dir / f"{n}.pairs.jsonl", pairs)
        write_jsonl(dpo_dir / f"{n}.jsonl", rows)
        hashes[n] = {
            "manifest_sha256": sha256_file(split_dir / f"{n}.manifest.jsonl"),
            "pairs_sha256": sha256_file(split_dir / f"{n}.pairs.jsonl"),
            "dpo_rows_sha256": sha256_file(dpo_dir / f"{n}.jsonl"),
        }
        report["splits"][n] = {
            "n_pairs": len(pairs),
            "n_dpo_rows": len(rows),
            "n_helpfulness_rows": sum(1 for r in rows if r["objective"] == "helpfulness"),
            "n_harmlessness_rows": sum(1 for r in rows if r["objective"] == "harmlessness"),
            "summary": audit.pool_summary(pairs),
        }
        assert len(rows) == 2 * len(pairs)
    report["manifest_hashes"] = hashes
    write_json(split_dir / "manifest_hashes.json", hashes)
    write_json(out_dir / "length_limits.json", {
        "max_prompt_length": max_prompt_length, "max_length": max_length, "tokenizer": cfg["tokenizer"],
        "source": source,
    })

    # 9. serialized examples + leakage test on every train row --------------------------------
    train_rows = data.pairs_to_dpo_rows(splits["train"])
    for r in train_rows:
        assert_no_objective_leakage(tok, r)
    examples: List[str] = []
    picked = []
    for t in data.PAIR_TYPES:
        pair = next(p for p in splits["train"] if p["pair_type"] == t)
        picked.extend(data.pair_to_dpo_rows(pair))
    for r in picked[: max(2, int(cfg.get("n_serialized_examples", 4)))]:
        examples.append(serialize_for_inspection(tok, r))
    (out_dir / "serialized_examples.txt").write_text("\n\n" + ("\n" + "=" * 100 + "\n\n").join(examples) + "\n", encoding="utf-8")

    write_json(out_dir / "data_audit.json", report)
    (out_dir / "data_audit.md").write_text(audit.render_markdown(report), encoding="utf-8")
    log.info("wrote %s", out_dir)
    return report


def load_split_rows(artifacts_dir: str, split: str) -> List[Dict[str, Any]]:
    from .utils import read_jsonl

    return read_jsonl(Path(artifacts_dir) / "dpo" / f"{split}.jsonl")


def load_length_limits(artifacts_dir: str) -> Dict[str, Any]:
    return read_json(Path(artifacts_dir) / "length_limits.json")
