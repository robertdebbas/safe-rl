"""Targeted shared-vs-separate diagnostic from EXISTING cross-scored eval responses (no training / generation).

Inputs: runs/diagnose_shared/scored_eval_{base,scalar,shared,separate}.jsonl (per response: n_tokens, truncated,
beaver_R/C, shared_R/C, separate_R/C, and scalar_S once the scalar RM has been applied), paired by prompt_hash
over the same 1,960 eval prompts; runs/grpo_*_seed42/rollout_batches.jsonl for the dynamics.
Outputs: runs/diagnose_shared/part2_report.json (+ a few plots). All CIs = paired bootstrap over prompts (2,000 reps).
Raw scores are never compared across scorers: everything is within-scorer differences, sign agreement, or ranks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "diagnose_shared"
METHODS = ["base", "scalar", "shared", "separate"]
RNG = np.random.default_rng(0)
NB = 2000


def load():
    data = {}
    for m in METHODS:
        rows = {r["prompt_hash"]: r for r in map(json.loads, open(OUT / f"scored_eval_{m}.jsonl"))}
        data[m] = rows
    common = sorted(set.intersection(*(set(v) for v in data.values())))
    cols = [k for k in next(iter(data["base"].values())) if k not in ("prompt_hash", "k")]
    arr = {m: {c: np.array([data[m][h][c] for h in common], float) for c in cols} for m in METHODS}
    return common, arr


def boot_mean(x):
    idx = RNG.integers(0, len(x), (NB, len(x)))
    b = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def reward_scorers(arr):
    return [c for c in ("scalar_S", "shared_R", "separate_R") if c in arr["base"]]


def part1(arr, report):
    """Pairwise ranking agreement with Beaver on identical prompts; over/under-valuation of each policy."""
    pairs = [("shared", "separate"), ("shared", "scalar"), ("shared", "base"), ("separate", "base"), ("scalar", "base"), ("separate", "scalar")]
    out = {}
    for a, b in pairs:
        res = {}
        for col, ext in [(c, "beaver_R") for c in reward_scorers(arr)] + [("shared_C", "beaver_C"), ("separate_C", "beaver_C")]:
            d_s = arr[a][col] - arr[b][col]; d_e = arr[a][ext] - arr[b][ext]
            res[col] = {"frac_scorer_prefers_a": boot_mean((d_s > 0).astype(float)),
                        "frac_beaver_prefers_a": boot_mean((d_e > 0).astype(float)),
                        "sign_agreement_with_beaver": boot_mean((np.sign(d_s) == np.sign(d_e)).astype(float)),
                        "spearman_of_differences": float(spearmanr(d_s, d_e).correlation)}
        out[f"{a}_vs_{b}"] = res
    report["part1_pairwise_agreement"] = out
    # over/under-valuation: pooled percentile rank of each policy's responses under each scorer vs its Beaver counterpart
    pool = {c: np.concatenate([arr[m][c] for m in METHODS]) for c in arr["base"]}
    n = len(arr["base"]["n_tokens"])
    ranks = {c: (np.argsort(np.argsort(pool[c])) / (len(pool[c]) - 1)) for c in pool}
    val = {}
    for i, m in enumerate(METHODS):
        sl = slice(i * n, (i + 1) * n)
        row = {}
        for col, ext in [(c, "beaver_R") for c in reward_scorers(arr)] + [("shared_C", "beaver_C"), ("separate_C", "beaver_C")]:
            diff = ranks[col][sl] - ranks[ext][sl]          # + : scorer ranks this policy's responses higher than Beaver does
            row[col] = {"mean_pooled_rank_scorer": float(ranks[col][sl].mean()), "mean_pooled_rank_beaver": float(ranks[ext][sl].mean()),
                        "rank_gap_scorer_minus_beaver": boot_mean(diff)}
        val[m] = row
    report["part1_pooled_rank_gaps"] = val
    # within-scorer standardized shift vs base (units = std of that scorer on base responses)
    std = {}
    for m in ("scalar", "shared", "separate"):
        std[m] = {c: boot_mean((arr[m][c] - arr["base"][c]) / arr["base"][c].std()) for c in arr["base"] if c not in ("n_tokens", "truncated")}
    report["part1_standardized_shift_vs_base"] = std


def part2(arr, report):
    """Length as explanation: length-matched shared vs scalar, reward-length relations, residual disagreement."""
    out = {}
    L = {m: arr[m]["n_tokens"] for m in METHODS}; T = {m: arr[m]["truncated"].astype(bool) for m in METHODS}
    both = T["shared"] & T["scalar"]
    out["shared_vs_scalar_both_truncated"] = {"n_prompts": int(both.sum()),
        "beaver_R_shared_minus_scalar": boot_mean(arr["shared"]["beaver_R"][both] - arr["scalar"]["beaver_R"][both]),
        "beaver_C_shared_minus_scalar": boot_mean(arr["shared"]["beaver_C"][both] - arr["scalar"]["beaver_C"][both]),
        "frac_beaver_C_shared_more_harmful": boot_mean((arr["shared"]["beaver_C"][both] > arr["scalar"]["beaver_C"][both]).astype(float)),
        "shared_C_shared_minus_scalar": boot_mean(arr["shared"]["shared_C"][both] - arr["scalar"]["shared_C"][both]),
        "separate_C_shared_minus_scalar": boot_mean(arr["shared"]["separate_C"][both] - arr["scalar"]["separate_C"][both])}
    nb = ~T["shared"] & ~T["scalar"]
    out["shared_vs_scalar_both_complete"] = {"n_prompts": int(nb.sum()),
        "beaver_C_shared_minus_scalar": boot_mean(arr["shared"]["beaver_C"][nb] - arr["scalar"]["beaver_C"][nb]) if nb.sum() > 5 else None}
    # shared vs separate length overlap
    bins = [0, 128, 256, 384, 511, 513]
    ov = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (L["shared"] >= lo) & (L["shared"] < hi) & (L["separate"] >= lo) & (L["separate"] < hi)
        row = {"bin": f"[{lo},{hi})", "n_prompts_both_in_bin": int(m.sum()), "n_shared_in_bin": int(((L['shared'] >= lo) & (L['shared'] < hi)).sum()),
               "n_separate_in_bin": int(((L['separate'] >= lo) & (L['separate'] < hi)).sum())}
        if m.sum() >= 20:
            row["beaver_C_shared_minus_separate"] = boot_mean(arr["shared"]["beaver_C"][m] - arr["separate"]["beaver_C"][m])
            row["beaver_R_shared_minus_separate"] = boot_mean(arr["shared"]["beaver_R"][m] - arr["separate"]["beaver_R"][m])
        ov.append(row)
    out["shared_vs_separate_length_overlap"] = ov
    # reward-length relation and reward-Beaver agreement per scorer, per method and pooled
    rel = {}
    for col, ext in [(c, "beaver_R") for c in reward_scorers(arr)] + [("shared_C", "beaver_C"), ("separate_C", "beaver_C"), ("beaver_R", None), ("beaver_C", None)]:
        r = {}
        for m in METHODS + ["pooled"]:
            x = np.concatenate([arr[k][col] for k in METHODS]) if m == "pooled" else arr[m][col]
            l = np.concatenate([L[k] for k in METHODS]) if m == "pooled" else L[m]
            e = (np.concatenate([arr[k][ext] for k in METHODS]) if m == "pooled" else arr[m][ext]) if ext else None
            r[m] = {"spearman_len": float(spearmanr(l, x).correlation)}
            if ext is not None:
                r[m]["spearman_with_beaver"] = float(spearmanr(x, e).correlation)
                # residualize both on (len, truncated) by OLS, then correlate the residuals
                t = (np.concatenate([T[k] for k in METHODS]) if m == "pooled" else T[m]).astype(float)
                X = np.column_stack([np.ones_like(l), l, t])
                rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]; re = e - X @ np.linalg.lstsq(X, e, rcond=None)[0]
                r[m]["spearman_with_beaver_after_length_control"] = float(spearmanr(rx, re).correlation)
        rel[col] = r
    out["reward_length_and_beaver_agreement"] = rel
    report["part2_length"] = out


def part3(report):
    out = {}
    for arm in ("shared", "separate"):
        R = sorted({r["batch_index"]: r for r in map(json.loads, open(ROOT / "runs" / f"grpo_{arm}_constrained_qwen2.5-1.5b_seed42" / "rollout_batches.jsonl"))}.values(), key=lambda r: r["batch_index"])
        g = lambda k: np.array([r[k] for r in R], float)
        lam, aR, aC = g("lambda_used"), g("adv_reward_abs_mean"), g("adv_cost_abs_mean")
        w_cost = lam * aC / (aR + lam * aC)            # share of the advantage magnitude coming from the cost term
        rew, cost, L, Tr, kl = g("reward_mean"), g("cost_batch_mean"), g("completion_len_mean"), g("truncation_rate"), g("kl_mean")
        viol = g("violation_ema_minus_d")
        def win(x, a, b): return float(x[a:b].mean())
        first_slack = int(np.argmax(lam < 0.2)) if (lam < 0.2).any() else None
        out[arm] = {"lambda_start_end": [float(lam[0]), float(lam[-1])], "first_batch_lambda_below_0.2": first_slack,
                    "effective_cost_share_of_advantage": {"batches_0_49": win(w_cost, 0, 50), "batches_100_149": win(w_cost, 100, 150), "batches_500_565": win(w_cost, 500, 566)},
                    "reward_mean": {"batches_0_49": win(rew, 0, 50), "batches_100_149": win(rew, 100, 150), "batches_500_565": win(rew, 500, 566)},
                    "cost_mean": {"batches_0_49": win(cost, 0, 50), "batches_100_149": win(cost, 100, 150), "batches_500_565": win(cost, 500, 566)},
                    "cost_min_batch_window50": int(np.argmin(np.convolve(cost, np.ones(50) / 50, "valid")) + 25),
                    "length": {"batches_0_49": win(L, 0, 50), "batches_100_149": win(L, 100, 150), "batches_500_565": win(L, 500, 566)},
                    "truncation": {"batches_0_49": win(Tr, 0, 50), "batches_100_149": win(Tr, 100, 150), "batches_500_565": win(Tr, 500, 566)},
                    "kl_end": float(kl[-20:].mean()), "frac_batches_violation_negative": float((viol < 0).mean()),
                    "spearman_after_slack": {k: float(spearmanr(np.arange(len(R))[150:], v[150:]).correlation) for k, v in (("reward", rew), ("cost", cost), ("length", L))}}
    report["part3_dynamics"] = out


def main():
    common, arr = load()
    report = {"n_prompts": len(common), "reward_scorers_available": reward_scorers(arr), "scalar_rm_scored": "scalar_S" in arr["base"]}
    part1(arr, report); part2(arr, report); part3(report)
    (OUT / "part2_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
