"""Data pipeline for Arm 1 (vanilla DPO on PKU-SafeRLHF).

Responsibilities:
  * load the OFFICIAL ``PKU-Alignment/PKU-SafeRLHF`` dataset at a pinned revision;
  * convert official rows into *underlying pair* records with stable ids;
  * validity filtering (per-reason counts, at the underlying-pair level);
  * prompt-leakage removal (normalized-prompt disjointness against the official test split);
  * deterministic stratified sampling of train / val / test_balanced / test_representative;
  * conversion of underlying pairs into DPO rows (exactly two per pair: helpfulness, harmlessness);
  * split manifests (ids + hashes + labels) for reuse by later arms.

Nothing in this module touches a model or tokenizer. Token-length filtering is driven by
``audit.py``, which hands back per-pair lengths; the exclusion itself happens here.
"""
from __future__ import annotations

import re
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .utils import get_logger, sha256_text

log = get_logger("saferl.data")

DATASET_NAME = "PKU-Alignment/PKU-SafeRLHF"

HARM_CATEGORIES = [
    "Endangering National Security",
    "Insulting Behavior",
    "Discriminatory Behavior",
    "Endangering Public Health",
    "Copyright Issues",
    "Violence",
    "Drugs",
    "Privacy Violation",
    "Economic Crime",
    "Mental Manipulation",
    "Human Trafficking",
    "Physical Harm",
    "Sexual Content",
    "Cybercrime",
    "Disrupting Public Order",
    "Environmental Damage",
    "Psychological Harm",
    "White-Collar Crime",
    "Animal Abuse",
]

SEVERITY_NAMES = {0: "none", 1: "minor", 2: "moderate", 3: "severe"}
OBJECTIVES = ("helpfulness", "harmlessness")
PAIR_TYPES = ("agreement", "conflict")
VALID_RESPONSE_IDS = (0, 1)

VALIDITY_REASONS = (
    "empty_prompt",
    "empty_response_0",
    "empty_response_1",
    "invalid_better_response_id",
    "invalid_safer_response_id",
)


# --------------------------------------------------------------------------- loading

def normalize_prompt(prompt: str) -> str:
    """Normalization used ONLY for leakage detection: strip + collapse whitespace."""
    return re.sub(r"\s+", " ", prompt.strip())


def load_official(revision: str, config: str = "default", name: str = DATASET_NAME):
    """Load the official dataset at a pinned revision. Returns (DatasetDict, metadata)."""
    import datasets as hf_datasets
    from datasets import load_dataset

    ds = load_dataset(name, config, revision=revision)
    meta = {
        "dataset": name,
        "config": config,
        "revision": revision,
        "datasets_version": hf_datasets.__version__,
        "official_split_sizes": {k: int(v.num_rows) for k, v in ds.items()},
        "schema": schema_dict(ds["train"].features),
    }
    return ds, meta


def schema_dict(features) -> Dict[str, Any]:
    """JSON-serializable rendering of a ``datasets.Features`` object."""

    def render(f):
        if hasattr(f, "dtype") and not hasattr(f, "feature") and not isinstance(f, dict):
            return str(f.dtype)
        if isinstance(f, dict):
            return {k: render(v) for k, v in f.items()}
        if hasattr(f, "feature"):
            return [render(f.feature)]
        return str(f)

    return {k: render(v) for k, v in features.items()}


def pair_id_for(official_split: str, official_index: int, prompt: str, r0: str, r1: str) -> str:
    return sha256_text(f"{official_split}|{official_index}|{prompt}|{r0}|{r1}")[:24]


def row_to_pair(row: Dict[str, Any], official_split: str, official_index: int) -> Dict[str, Any]:
    """Convert one official row into an underlying-pair record. Does NOT filter."""
    prompt = row["prompt"] if row["prompt"] is not None else ""
    r0 = row["response_0"] if row["response_0"] is not None else ""
    r1 = row["response_1"] if row["response_1"] is not None else ""
    better = row["better_response_id"]
    safer = row["safer_response_id"]
    key = normalize_prompt(prompt)

    def cats(d: Any) -> List[str]:
        if not isinstance(d, dict):
            return []
        return [c for c in HARM_CATEGORIES if bool(d.get(c, False))]

    sev0 = int(row["response_0_severity_level"]) if row["response_0_severity_level"] is not None else -1
    sev1 = int(row["response_1_severity_level"]) if row["response_1_severity_level"] is not None else -1
    safe0 = bool(row["is_response_0_safe"])
    safe1 = bool(row["is_response_1_safe"])
    better_ok = better in VALID_RESPONSE_IDS
    safer_ok = safer in VALID_RESPONSE_IDS
    pair_type = None
    if better_ok and safer_ok:
        pair_type = "agreement" if better == safer else "conflict"
    return OrderedDict(
        pair_id=pair_id_for(official_split, official_index, prompt, r0, r1),
        official_split=official_split,
        official_index=int(official_index),
        prompt=prompt,
        response_0=r0,
        response_1=r1,
        prompt_key=key,
        prompt_hash=sha256_text(key)[:16],
        prompt_source=row.get("prompt_source"),
        response_0_source=row.get("response_0_source"),
        response_1_source=row.get("response_1_source"),
        is_response_0_safe=safe0,
        is_response_1_safe=safe1,
        response_0_severity_level=sev0,
        response_1_severity_level=sev1,
        response_0_harm_category=cats(row.get("response_0_harm_category")),
        response_1_harm_category=cats(row.get("response_1_harm_category")),
        better_response_id=int(better) if better is not None else None,
        safer_response_id=int(safer) if safer is not None else None,
        response_0_sha256=row.get("response_0_sha256"),
        response_1_sha256=row.get("response_1_sha256"),
        pair_type=pair_type,
        max_severity=max(sev0, sev1),
        safety_pattern=f"{'safe' if safe0 else 'unsafe'},{'safe' if safe1 else 'unsafe'}",
    )


def dataset_to_pairs(dataset, official_split: str) -> List[Dict[str, Any]]:
    cols = dataset.column_names
    out = []
    # Column-wise access is far faster than row-wise on arrow-backed datasets.
    columns = {c: dataset[c] for c in cols}
    n = dataset.num_rows
    for i in range(n):
        row = {c: columns[c][i] for c in cols}
        out.append(row_to_pair(row, official_split, i))
    return out


# --------------------------------------------------------------------------- validity

def validity_reasons(pair: Dict[str, Any]) -> List[str]:
    reasons = []
    if not pair["prompt"].strip():
        reasons.append("empty_prompt")
    if not pair["response_0"].strip():
        reasons.append("empty_response_0")
    if not pair["response_1"].strip():
        reasons.append("empty_response_1")
    if pair["better_response_id"] not in VALID_RESPONSE_IDS:
        reasons.append("invalid_better_response_id")
    if pair["safer_response_id"] not in VALID_RESPONSE_IDS:
        reasons.append("invalid_safer_response_id")
    return reasons


def filter_valid(pairs: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Drop invalid underlying pairs. Returns (kept, counts). A pair failing several checks is
    counted under every reason it fails and once in ``removed_total``."""
    kept = []
    counts: Dict[str, Any] = {r: 0 for r in VALIDITY_REASONS}
    removed_total = 0
    removed_multi = 0
    for p in pairs:
        reasons = validity_reasons(p)
        if reasons:
            removed_total += 1
            if len(reasons) > 1:
                removed_multi += 1
            for r in reasons:
                counts[r] += 1
        else:
            kept.append(p)
    counts["removed_total"] = removed_total
    counts["removed_with_multiple_reasons"] = removed_multi
    counts["input"] = len(pairs)
    counts["kept"] = len(kept)
    return kept, counts


def agreement_stats(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(pairs)
    agree = sum(1 for p in pairs if p["pair_type"] == "agreement")
    conflict = sum(1 for p in pairs if p["pair_type"] == "conflict")
    unknown = n - agree - conflict
    return {
        "n": n,
        "agreement": agree,
        "conflict": conflict,
        "undetermined": unknown,
        "agreement_pct": (100.0 * agree / n) if n else None,
        "conflict_pct": (100.0 * conflict / n) if n else None,
    }


# --------------------------------------------------------------------------- leakage

def remove_prompt_leakage(
    pairs: Sequence[Dict[str, Any]], forbidden_prompt_keys: Set[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept = [p for p in pairs if p["prompt_key"] not in forbidden_prompt_keys]
    removed = [p for p in pairs if p["prompt_key"] in forbidden_prompt_keys]
    return kept, {
        "removed_pairs": len(removed),
        "removed_unique_prompts": len({p["prompt_key"] for p in removed}),
        "kept_pairs": len(kept),
    }


def apply_length_limits(
    pairs: Sequence[Dict[str, Any]],
    lengths: Dict[str, Dict[str, int]],
    max_prompt_length: int,
    max_length: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Exclude underlying pairs whose serialized prompt exceeds ``max_prompt_length`` or whose
    prompt+response_0 / prompt+response_1 exceeds ``max_length``. ``lengths[pair_id]`` must have
    keys ``prompt``, ``total_0``, ``total_1`` (computed by audit.py with the real tokenizer)."""
    kept, n_prompt, n_total, n_any = [], 0, 0, 0
    for p in pairs:
        L = lengths[p["pair_id"]]
        over_prompt = L["prompt"] > max_prompt_length
        over_total = max(L["total_0"], L["total_1"]) > max_length
        if over_prompt:
            n_prompt += 1
        if over_total:
            n_total += 1
        if over_prompt or over_total:
            n_any += 1
        else:
            kept.append(p)
    return kept, {
        "input": len(pairs),
        "removed_over_prompt_limit": n_prompt,
        "removed_over_total_limit": n_total,
        "removed_total": n_any,
        "kept": len(kept),
    }


# --------------------------------------------------------------------------- sampling

def stratum_of(pair: Dict[str, Any]) -> str:
    return f"{pair['prompt_source']}|sev{pair['max_severity']}|{pair['safety_pattern']}"


def largest_remainder(weights: Dict[str, float], total: int) -> Dict[str, int]:
    """Hamilton apportionment of ``total`` across keys proportional to ``weights``."""
    keys = sorted(weights)
    w = np.array([max(0.0, float(weights[k])) for k in keys], dtype=float)
    if total <= 0 or w.sum() <= 0:
        return {k: 0 for k in keys}
    exact = w / w.sum() * total
    floors = np.floor(exact).astype(int)
    rem = total - int(floors.sum())
    frac = exact - floors
    # deterministic tie-break: larger fractional part first, then key order
    order = sorted(range(len(keys)), key=lambda i: (-frac[i], keys[i]))
    for i in order[:rem]:
        floors[i] += 1
    return {k: int(v) for k, v in zip(keys, floors)}


def stratified_sample(
    candidates: Sequence[Dict[str, Any]],
    n: int,
    rng: np.random.Generator,
    reference: Optional[Sequence[Dict[str, Any]]] = None,
    max_pairs_per_prompt: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deterministic proportional stratified sampling of ``n`` pairs from ``candidates``.

    Strata = (prompt_source, max_severity, safety_pattern). Target allocation is proportional to
    the stratum frequencies in ``reference`` (defaults to ``candidates``), so a split sampled
    from a reduced pool still mirrors the full pool's composition. If a stratum has fewer
    available candidates than allocated, the deficit is redistributed to the other strata
    proportionally to their remaining availability; everything is reported.
    """
    reference = candidates if reference is None else reference
    ref_counts = Counter(stratum_of(p) for p in reference)
    by_stratum: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in candidates:
        by_stratum[stratum_of(p)].append(p)
    # Available counts can only come from candidates; strata absent from reference get weight 0
    # but still may absorb redistributed deficit.
    alloc = largest_remainder({s: ref_counts.get(s, 0) for s in set(ref_counts) | set(by_stratum)}, n)
    avail = {s: len(v) for s, v in by_stratum.items()}
    taken = {s: min(alloc.get(s, 0), avail.get(s, 0)) for s in alloc}
    deficit = n - sum(taken.values())
    redistribution_rounds = 0
    while deficit > 0:
        room = {s: avail.get(s, 0) - taken[s] for s in alloc if avail.get(s, 0) - taken[s] > 0}
        if not room:
            break
        extra = largest_remainder(room, min(deficit, sum(room.values())))
        for s, k in extra.items():
            taken[s] += min(k, room[s])
        deficit = n - sum(taken.values())
        redistribution_rounds += 1
    if deficit > 0:
        raise ValueError(f"Cannot sample {n} pairs: only {n - deficit} candidates available")

    selected: List[Dict[str, Any]] = []
    per_prompt: Counter = Counter()
    short_strata: Dict[str, Any] = {}
    for s in sorted(taken):
        k = taken[s]
        if k == 0:
            continue
        pool = sorted(by_stratum[s], key=lambda p: p["pair_id"])  # order-independent determinism
        perm = rng.permutation(len(pool))
        got = 0
        for idx in perm:
            if got >= k:
                break
            p = pool[idx]
            if max_pairs_per_prompt is not None and per_prompt[p["prompt_key"]] >= max_pairs_per_prompt:
                continue
            selected.append(p)
            per_prompt[p["prompt_key"]] += 1
            got += 1
        if got < k:
            short_strata[s] = {"wanted": k, "got": got}
    if len(selected) < n:
        # per-prompt cap starved some strata: top up from any stratum with room, deterministically
        chosen_ids = {p["pair_id"] for p in selected}
        leftovers = sorted((p for p in candidates if p["pair_id"] not in chosen_ids), key=lambda p: p["pair_id"])
        perm = rng.permutation(len(leftovers))
        for idx in perm:
            if len(selected) >= n:
                break
            p = leftovers[idx]
            if max_pairs_per_prompt is not None and per_prompt[p["prompt_key"]] >= max_pairs_per_prompt:
                continue
            selected.append(p)
            per_prompt[p["prompt_key"]] += 1
        if len(selected) < n:
            raise ValueError(f"Cannot sample {n} pairs under max_pairs_per_prompt={max_pairs_per_prompt}")
    selected.sort(key=lambda p: p["pair_id"])
    report = {
        "n_requested": n,
        "n_selected": len(selected),
        "n_strata": len(alloc),
        "redistribution_rounds": redistribution_rounds,
        "strata_with_deficit": {s: {"allocated": alloc[s], "available": avail.get(s, 0), "taken": taken[s]}
                                for s in sorted(alloc) if alloc[s] > avail.get(s, 0)},
        "strata_short_due_to_prompt_cap": short_strata,
        "allocation": {s: {"reference": ref_counts.get(s, 0), "allocated": alloc[s],
                           "available": avail.get(s, 0), "taken": taken[s]} for s in sorted(alloc)},
    }
    return selected, report


def sample_split(
    pool: Sequence[Dict[str, Any]],
    quotas: Dict[str, int],
    rng: np.random.Generator,
    owned_prompts: Set[str],
    reference_pool: Optional[Sequence[Dict[str, Any]]] = None,
    max_pairs_per_prompt: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Sample a split with exact per-pair-type quotas from ``pool`` excluding prompts already
    owned by another split. Every prompt used is then owned by this split."""
    reference_pool = pool if reference_pool is None else reference_pool
    selected, report = [], {}
    for pair_type in PAIR_TYPES:
        n = int(quotas.get(pair_type, 0))
        if n == 0:
            continue
        cands = [p for p in pool if p["pair_type"] == pair_type and p["prompt_key"] not in owned_prompts]
        ref = [p for p in reference_pool if p["pair_type"] == pair_type]
        sel, rep = stratified_sample(cands, n, rng, reference=ref, max_pairs_per_prompt=max_pairs_per_prompt)
        rep["n_candidates_after_prompt_exclusion"] = len(cands)
        report[pair_type] = rep
        selected.extend(sel)
    selected.sort(key=lambda p: p["pair_id"])
    assert len({p["pair_id"] for p in selected}) == len(selected)
    return selected, report


def natural_quotas(reference_pool: Sequence[Dict[str, Any]], total: int) -> Dict[str, int]:
    """Agreement/conflict quotas that mirror the natural distribution of ``reference_pool``."""
    c = Counter(p["pair_type"] for p in reference_pool)
    return largest_remainder({t: c.get(t, 0) for t in PAIR_TYPES}, total)


def build_splits(
    train_pool: Sequence[Dict[str, Any]],
    test_pool: Sequence[Dict[str, Any]],
    sizes: Dict[str, Dict[str, int]],
    data_seed: int,
    max_pairs_per_prompt: Optional[int] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Deterministic construction of the four splits.

    Order (each later split only uses prompts not owned by the earlier ones of the same source):
      1. train             <- official train pool  (quotas sizes['train'])
      2. val               <- official train pool, prompts disjoint from train
      3. test_balanced     <- official test pool    (quotas sizes['test_balanced'])
      4. test_representative <- official test pool, prompts disjoint from test_balanced,
                               quotas = natural agreement/conflict split of the eligible test pool.
    ``test_balanced`` is sampled first because it is the binding constraint on conflict pairs.
    """
    rng = np.random.default_rng(data_seed)
    splits: Dict[str, List[Dict[str, Any]]] = {}
    report: Dict[str, Any] = {"data_seed": data_seed, "order": []}

    owned_train: Set[str] = set()
    for name in ("train", "val"):
        q = sizes[name]
        sel, rep = sample_split(train_pool, q, rng, owned_train, reference_pool=train_pool,
                                max_pairs_per_prompt=max_pairs_per_prompt)
        splits[name] = sel
        owned_train |= {p["prompt_key"] for p in sel}
        report[name] = {"quotas": q, "sampling": rep}
        report["order"].append(name)

    owned_test: Set[str] = set()
    q_bal = sizes["test_balanced"]
    sel, rep = sample_split(test_pool, q_bal, rng, owned_test, reference_pool=test_pool,
                            max_pairs_per_prompt=max_pairs_per_prompt)
    splits["test_balanced"] = sel
    owned_test |= {p["prompt_key"] for p in sel}
    report["test_balanced"] = {"quotas": q_bal, "sampling": rep}
    report["order"].append("test_balanced")

    total_rep = int(sizes["test_representative"]["total"])
    q_rep = natural_quotas(test_pool, total_rep)
    sel, rep = sample_split(test_pool, q_rep, rng, owned_test, reference_pool=test_pool,
                            max_pairs_per_prompt=max_pairs_per_prompt)
    splits["test_representative"] = sel
    owned_test |= {p["prompt_key"] for p in sel}
    report["test_representative"] = {
        "quotas": q_rep,
        "natural_conflict_fraction_of_eligible_test_pool": agreement_stats(test_pool)["conflict_pct"],
        "sampling": rep,
    }
    report["order"].append("test_representative")
    return splits, report


# --------------------------------------------------------------------------- DPO rows

def pair_to_dpo_rows(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Exactly two DPO rows per underlying pair. ``objective`` is METADATA ONLY and must never be
    serialized into the model input (see formatting.py and tests)."""
    r = [pair["response_0"], pair["response_1"]]
    rows = []
    for objective, pref_id in (("helpfulness", pair["better_response_id"]),
                               ("harmlessness", pair["safer_response_id"])):
        rows.append(OrderedDict(
            row_id=f"{pair['pair_id']}:{'H' if objective == 'helpfulness' else 'S'}",
            pair_id=pair["pair_id"],
            objective=objective,
            pair_type=pair["pair_type"],
            prompt=pair["prompt"],
            chosen=r[pref_id],
            rejected=r[1 - pref_id],
            chosen_id=int(pref_id),
            rejected_id=int(1 - pref_id),
            better_response_id=pair["better_response_id"],
            safer_response_id=pair["safer_response_id"],
        ))
    return rows


def pairs_to_dpo_rows(pairs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in pairs:
        out.extend(pair_to_dpo_rows(p))
    return out


# --------------------------------------------------------------------------- manifests

MANIFEST_FIELDS = (
    "pair_id", "official_split", "official_index", "prompt_hash", "prompt_source",
    "response_0_source", "response_1_source", "response_0_sha256", "response_1_sha256",
    "is_response_0_safe", "is_response_1_safe", "response_0_severity_level",
    "response_1_severity_level", "response_0_harm_category", "response_1_harm_category",
    "better_response_id", "safer_response_id", "pair_type", "max_severity", "safety_pattern",
)


def manifest_record(pair: Dict[str, Any]) -> Dict[str, Any]:
    rec = {k: pair[k] for k in MANIFEST_FIELDS}
    rec["prompt_sha256"] = sha256_text(pair["prompt"])
    rec["response_0_sha256_local"] = sha256_text(pair["response_0"])
    rec["response_1_sha256_local"] = sha256_text(pair["response_1"])
    return rec


def check_prompt_disjoint(splits: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, int]:
    """Pairwise normalized-prompt overlap counts between splits (all must be 0)."""
    keys = {name: {p["prompt_key"] for p in pairs} for name, pairs in splits.items()}
    names = sorted(keys)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}&{b}"] = len(keys[a] & keys[b])
    return out
