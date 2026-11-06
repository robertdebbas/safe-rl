"""Dataset audit: token-length quantiles, threshold selection, distribution summaries, and the
Markdown rendering of the data audit report."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

from .data import HARM_CATEGORIES, PAIR_TYPES, SEVERITY_NAMES, agreement_stats
from .formatting import prompt_text
from .utils import percentile_stats


# --------------------------------------------------------------------------- token lengths

def compute_lengths(tokenizer, pairs: Sequence[Dict[str, Any]], batch_size: int = 2048) -> Dict[str, Dict[str, int]]:
    """Serialized token lengths per pair, using exactly the formatting used in training:
    prompt = chat-templated user turn (+assistant header); completion = response + EOS."""
    out: Dict[str, Dict[str, int]] = {}
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        p_texts = [prompt_text(tokenizer, p["prompt"]) for p in chunk]
        p_len = [len(x) for x in tokenizer(p_texts, add_special_tokens=False)["input_ids"]]
        r0_len = [len(x) + 1 for x in tokenizer([p["response_0"] for p in chunk], add_special_tokens=False)["input_ids"]]
        r1_len = [len(x) + 1 for x in tokenizer([p["response_1"] for p in chunk], add_special_tokens=False)["input_ids"]]
        for p, a, b, c in zip(chunk, p_len, r0_len, r1_len):
            out[p["pair_id"]] = {"prompt": a, "response_0": b, "response_1": c,
                                 "total_0": a + b, "total_1": a + c}
    return out


def length_quantiles(lengths: Dict[str, Dict[str, int]], pair_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    ids = list(lengths) if pair_ids is None else list(pair_ids)
    prompt = [lengths[i]["prompt"] for i in ids]
    t0 = [lengths[i]["total_0"] for i in ids]
    t1 = [lengths[i]["total_1"] for i in ids]
    tmax = [max(a, b) for a, b in zip(t0, t1)]
    return {
        "prompt": percentile_stats(prompt),
        "prompt_plus_response_0": percentile_stats(t0),
        "prompt_plus_response_1": percentile_stats(t1),
        "prompt_plus_response_both_sequences": percentile_stats(t0 + t1),
        "pair_max_total": percentile_stats(tmax),
    }


def coverage(values: Sequence[int], candidates: Sequence[int]) -> Dict[str, float]:
    n = max(1, len(values))
    return {str(c): sum(1 for v in values if v <= c) / n for c in candidates}


def choose_threshold(values: Sequence[int], candidates: Sequence[int], target: float) -> Optional[int]:
    """Smallest candidate whose coverage >= target, or None if no candidate reaches it."""
    cov = coverage(values, candidates)
    for c in sorted(candidates):
        if cov[str(c)] >= target:
            return int(c)
    return None


# --------------------------------------------------------------------------- distributions

def _dist(counter: Counter, total: Optional[int] = None) -> Dict[str, Any]:
    total = sum(counter.values()) if total is None else total
    return {str(k): {"count": int(v), "pct": (100.0 * v / total) if total else None}
            for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def harm_category_marginal(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pair_level = Counter()
    response_level = Counter()
    for p in pairs:
        c0, c1 = set(p["response_0_harm_category"]), set(p["response_1_harm_category"])
        for c in c0 | c1:
            pair_level[c] += 1
        for c in c0:
            response_level[c] += 1
        for c in c1:
            response_level[c] += 1
    n = len(pairs)
    return {
        "pair_level_any_response": {c: {"count": pair_level.get(c, 0), "pct": (100.0 * pair_level.get(c, 0) / n) if n else None}
                                    for c in HARM_CATEGORIES},
        "response_level": {c: {"count": response_level.get(c, 0), "pct": (100.0 * response_level.get(c, 0) / (2 * n)) if n else None}
                           for c in HARM_CATEGORIES},
        "pairs_with_no_category": sum(1 for p in pairs if not p["response_0_harm_category"] and not p["response_1_harm_category"]),
    }


def pool_summary(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """All distributions requested by the audit, overall and per pair type."""
    def one(ps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        per_prompt = Counter(p["prompt_key"] for p in ps)
        return {
            "n_pairs": len(ps),
            "n_unique_prompts": len(per_prompt),
            "max_pairs_per_prompt": max(per_prompt.values()) if per_prompt else 0,
            "prompt_source": _dist(Counter(p["prompt_source"] for p in ps)),
            "response_source_pair": _dist(Counter(f"{p['response_0_source']}|{p['response_1_source']}" for p in ps)),
            "max_severity": _dist(Counter(f"{p['max_severity']}:{SEVERITY_NAMES.get(p['max_severity'], '?')}" for p in ps)),
            "severity_response_0": _dist(Counter(p["response_0_severity_level"] for p in ps)),
            "severity_response_1": _dist(Counter(p["response_1_severity_level"] for p in ps)),
            "safety_pattern": _dist(Counter(p["safety_pattern"] for p in ps)),
            "better_response_id": _dist(Counter(p["better_response_id"] for p in ps)),
            "safer_response_id": _dist(Counter(p["safer_response_id"] for p in ps)),
            "harm_categories": harm_category_marginal(ps),
        }

    out = {"agreement_conflict": agreement_stats(pairs), "all": one(pairs)}
    for t in PAIR_TYPES:
        out[t] = one([p for p in pairs if p["pair_type"] == t])
    return out


# --------------------------------------------------------------------------- markdown

def _fmt_pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.2f}%"


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def _quant_rows(name: str, q: Dict[str, Any]) -> List[Any]:
    return [name, q.get("n"), q.get("p50"), q.get("p90"), q.get("p95"), q.get("p99"), q.get("max")]


def render_markdown(report: Dict[str, Any]) -> str:
    md: List[str] = []
    src = report["source"]
    md.append("# PKU-SafeRLHF data audit (Arm 1: vanilla DPO)\n")
    md.append(f"- dataset: `{src['dataset']}` config `{src['config']}` revision `{src['revision']}` (datasets {src['datasets_version']})")
    md.append(f"- tokenizer: `{report['tokenizer']}`")
    md.append(f"- data_seed: {report['data_seed']}")
    md.append(f"- official split sizes: {src['official_split_sizes']}\n")

    md.append("## Schema (observed)\n")
    md.append("```json")
    import json
    md.append(json.dumps(src["schema"], indent=1))
    md.append("```\n")

    md.append("## Agreement / conflict on the OFFICIAL source (all rows with valid ids)\n")
    rows = []
    for split, st in report["official_agreement"].items():
        rows.append([split, st["n"], st["agreement"], _fmt_pct(st["agreement_pct"]), st["conflict"], _fmt_pct(st["conflict_pct"]), st["undetermined"]])
    md.append(_table(["split", "n", "agreement", "agree %", "conflict", "conflict %", "undetermined"], rows) + "\n")

    md.append("## Validity filtering (underlying-pair level)\n")
    rows = []
    for split, c in report["validity"].items():
        rows.append([split, c["input"], c["empty_prompt"], c["empty_response_0"], c["empty_response_1"],
                     c["invalid_better_response_id"], c["invalid_safer_response_id"], c["removed_total"], c["kept"]])
    md.append(_table(["split", "input", "empty prompt", "empty r0", "empty r1", "bad better_id", "bad safer_id", "removed (any)", "kept"], rows) + "\n")

    lk = report["leakage"]
    md.append("## Prompt leakage between OFFICIAL train and test\n")
    md.append(f"- normalized prompts shared by official train and official test: **{lk['shared_normalized_prompts']}** "
              f"(of {lk['unique_train_prompts']} unique train / {lk['unique_test_prompts']} unique test prompts)")
    md.append(f"- policy: drop every official-train pair whose normalized prompt occurs anywhere in the official test split")
    md.append(f"- train pairs removed: **{lk['removed_pairs']}** ({lk['removed_unique_prompts']} prompts); train pool kept: {lk['kept_pairs']}\n")

    md.append("## Token lengths (eligible candidate pool after validity + leakage filtering)\n")
    for pool_name, q in report["lengths"]["quantiles"].items():
        md.append(f"### {pool_name}\n")
        rows = [_quant_rows(k, v) for k, v in q.items()]
        md.append(_table(["sequence", "n", "p50", "p90", "p95", "p99", "max"], rows) + "\n")
    md.append("### Coverage by candidate threshold (combined eligible pool)\n")
    md.append("prompt: " + ", ".join(f"{k}: {100*v:.3f}%" for k, v in report["lengths"]["coverage"]["prompt"].items()))
    md.append("\ntotal (both sequences): " + ", ".join(f"{k}: {100*v:.3f}%" for k, v in report["lengths"]["coverage"]["total"].items()) + "\n")
    lim = report["length_limits"]
    md.append(f"- selected `max_prompt_length` = **{lim['max_prompt_length']}** (source: {lim['source']})")
    md.append(f"- selected `max_length` = **{lim['max_length']}** (source: {lim['source']})")
    md.append(f"- coverage target: {lim['coverage_target']}; prompt coverage {100*lim['prompt_coverage']:.3f}%, total coverage {100*lim['total_coverage']:.3f}%; "
              f"both >= target: {lim['coverage_target_met']}")
    md.append(f"- underlying pairs excluded for length (train pool / test pool): {report['length_exclusion']['train']['removed_total']} / "
              f"{report['length_exclusion']['test']['removed_total']} "
              f"({100*report['length_exclusion']['fraction_excluded_combined']:.3f}% of the combined eligible pool)\n")

    md.append("## Final split sizes\n")
    rows = []
    for name, s in report["splits"].items():
        ac = s["summary"]["agreement_conflict"]
        rows.append([name, ac["n"], ac["agreement"], ac["conflict"], s["summary"]["all"]["n_unique_prompts"],
                     s["summary"]["all"]["max_pairs_per_prompt"], s["n_dpo_rows"], s["n_helpfulness_rows"], s["n_harmlessness_rows"]])
    md.append(_table(["split", "pairs", "agreement", "conflict", "unique prompts", "max pairs/prompt", "DPO rows", "H rows", "S rows"], rows) + "\n")
    md.append("### Sampling notes\n")
    for name, s in report["splits"].items():
        samp = report["sampling_report"][name]
        md.append(f"- **{name}**: quotas {samp['quotas']}")
        for t, rep in samp["sampling"].items():
            md.append(f"  - {t}: candidates after prompt exclusion {rep['n_candidates_after_prompt_exclusion']}, "
                      f"strata {rep['n_strata']}, redistribution rounds {rep['redistribution_rounds']}, "
                      f"strata with deficit {len(rep['strata_with_deficit'])}")
    md.append("")

    md.append("## Prompt disjointness (normalized prompt overlap counts; must all be 0)\n")
    md.append("```")
    md.append(json.dumps(report["prompt_disjointness"], indent=1))
    md.append("```")
    md.append(f"- official test rows inside train/val: {report['official_test_in_train_val']}\n")

    md.append("## Distributions before vs after sampling\n")
    pools = {"eligible_train_pool": report["pools"]["train"], "eligible_test_pool": report["pools"]["test"]}
    pools.update({k: v["summary"] for k, v in report["splits"].items()})
    for dist_name in ("prompt_source", "max_severity", "safety_pattern"):
        md.append(f"### {dist_name}\n")
        keys = sorted({k for s in pools.values() for k in s["all"][dist_name]})
        rows = [[k] + [_fmt_pct(s["all"][dist_name].get(k, {}).get("pct")) for s in pools.values()] for k in keys]
        md.append(_table([dist_name] + list(pools), rows) + "\n")
        for t in PAIR_TYPES:
            rows = [[k] + [_fmt_pct(s[t][dist_name].get(k, {}).get("pct")) for s in pools.values()] for k in keys]
            md.append(f"*{t} pairs only*\n\n" + _table([dist_name] + list(pools), rows) + "\n")
    md.append("### Harm categories (pair-level marginal: % of pairs where either response carries the category)\n")
    rows = [[c] + [_fmt_pct(s["all"]["harm_categories"]["pair_level_any_response"][c]["pct"]) for s in pools.values()] for c in HARM_CATEGORIES]
    md.append(_table(["category"] + list(pools), rows) + "\n")

    md.append("## Manifest hashes (sha256)\n")
    md.append("```")
    md.append(json.dumps(report["manifest_hashes"], indent=1))
    md.append("```\n")
    return "\n".join(md)
