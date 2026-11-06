"""Post-hoc diagnosis of the shared constrained-GRPO result from EXISTING artifacts only (no training, no setting changes).

Stage ``score`` (GPU): re-scores already-saved response sets with every frozen scorer so they can be compared on the same text:
    shared RM (reward + cost heads), separate R and C models (RM seed 42, best-by-val, the GRPO scorers),
    and the external Beaver reward/cost evaluators (reused from runs/eval when already computed).
  Response sets (text that exists on disk; per-checkpoint monitoring text was never saved):
    pi0_calibration   artifacts/grpo/calibration_generations.jsonl      initial policy, 256 calibration prompts x 4
    shared_b100       runs/grpo_shared_..._seed42/diagnostic/batch_00100_samples.jsonl    (32 monitoring prompts)
    separate_b100     runs/grpo_separate_..._seed42/diagnostic/batch_00100_samples.jsonl  (32 monitoring prompts)
    eval_<method>     runs/eval/<method>/seed{0|42}/responses.jsonl  final checkpoints on the 1,960 eval prompts (base = pi0)
Stage ``analyze`` (CPU): RM validation comparison, dual-update audit, scorer agreement, length/truncation relationships, plots.
Outputs go to runs/diagnose_shared/ only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "runs" / "diagnose_shared"
RUNS = {"shared": ROOT / "runs" / "grpo_shared_constrained_qwen2.5-1.5b_seed42",
        "separate": ROOT / "runs" / "grpo_separate_constrained_qwen2.5-1.5b_seed42",
        "scalar": ROOT / "runs" / "grpo_scalar_qwen2.5-1.5b_seed42"}
RM_RUNS = {"shared": ROOT / "runs" / "rm_shared_qwen2.5-1.5b_seed42",
           "separate_reward": ROOT / "runs" / "rm_separate_reward_qwen2.5-1.5b_seed42",
           "separate_cost": ROOT / "runs" / "rm_separate_cost_qwen2.5-1.5b_seed42"}
EVAL_SETS = {"eval_base": ("base", 0), "eval_scalar": ("grpo_scalar", 42), "eval_shared": ("grpo_shared", 42), "eval_separate": ("grpo_separate", 42)}


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ----------------------------------------------------------------------------------------------- stage: score
def load_sets() -> Dict[str, List[Dict[str, Any]]]:
    sets: Dict[str, List[Dict[str, Any]]] = {}
    cal = read_jsonl(ROOT / "artifacts" / "grpo" / "calibration_generations.jsonl")
    sets["pi0_calibration"] = [{"prompt_hash": r["prompt_hash"], "k": r["k"], "prompt": r["prompt"], "response": r["response"],
                                "n_tokens": r["n_completion_tokens"], "truncated": r["truncated"]} for r in cal]
    for arm in ("shared", "separate"):
        p = RUNS[arm] / "diagnostic" / "batch_00100_samples.jsonl"
        if p.exists():
            sets[f"{arm}_b100"] = [{"prompt_hash": r["prompt_hash"], "k": 0, "prompt": r["prompt"], "response": r["response"],
                                    "n_tokens": r["n_completion_tokens"], "truncated": r["truncated"]} for r in read_jsonl(p)]
    for name, (method, seed) in EVAL_SETS.items():
        d = ROOT / "runs" / "eval" / method / f"seed{seed}"
        if not (d / "responses.jsonl").exists():
            continue
        sc = {r["prompt_hash"]: r for r in read_jsonl(d / "scores.jsonl")}
        sets[name] = [{"prompt_hash": r["prompt_hash"], "k": 0, "prompt": r["prompt"], "response": r["response"],
                       "n_tokens": r["n_completion_tokens"], "truncated": r["truncated"],
                       "beaver_R": sc[r["prompt_hash"]]["reward_score"], "beaver_C": sc[r["prompt_hash"]]["cost_score"]}
                      for r in read_jsonl(d / "responses.jsonl")]
    return sets


def stage_score() -> None:
    import torch
    from saferl.utils import load_config
    from saferl.eval_score import EVALUATOR_TEMPLATE, PKUScoreModel, load_evaluator_tokenizer, score_pairs
    from saferl.grpo_scoring import FrozenScorers
    from saferl.modeling import load_tokenizer

    gcfg = load_config(str(ROOT / "configs" / "grpo_common.yaml"), [])
    ecfg = load_config(str(ROOT / "configs" / "eval.yaml"), [])
    gen = gcfg["generation"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets = load_sets()
    print({k: len(v) for k, v in sets.items()}, flush=True)
    tok = load_tokenizer(gcfg["policy_model"])
    ckpts = {"shared": {"shared": str(RM_RUNS["shared"] / "checkpoints" / "best")},
             "separate": {"reward": str(RM_RUNS["separate_reward"] / "checkpoints" / "best"),
                          "cost": str(RM_RUNS["separate_cost"] / "checkpoints" / "best")}}
    for arm in ("shared", "separate"):   # identical scorer construction to train_grpo / grpo_calibrate
        sc = FrozenScorers(arm, ckpts[arm], device, gcfg["precision"], int(gcfg.get("scorer_batch_size", 16)),
                           gen["max_prompt_length"], gen["max_prompt_length"] + gen["max_completion_length"])
        for name, rows in sets.items():
            out = sc.score(tok, [r["prompt"] for r in rows], [r["response"] for r in rows])
            for i, r in enumerate(rows):
                r[f"{arm}_R"] = float(out["reward"][i]); r[f"{arm}_C"] = float(out["cost"][i])
            print(f"{arm} scored {name}", flush=True)
        del sc
        torch.cuda.empty_cache()
    ev = ecfg["evaluators"]
    for key, col in (("reward", "beaver_R"), ("cost", "beaver_C")):
        need = {n: rows for n, rows in sets.items() if col not in rows[0]}
        if not need:
            continue
        etok = load_evaluator_tokenizer(ev[key]["model"], ev[key].get("revision"))
        m = PKUScoreModel.from_pretrained(ev[key]["model"], device, torch.bfloat16, ev[key].get("revision"))
        for name, rows in need.items():
            res = score_pairs(m, etok, [r["prompt"] for r in rows], [r["response"] for r in rows], device, ecfg["precision"],
                              ev.get("template", EVALUATOR_TEMPLATE), bool(ev.get("append_eos", True)), int(ev["max_length"]), int(ev["batch_size"]))
            for r, s in zip(rows, res["scores"]):
                r[col] = float(s)
            print(f"beaver {key} scored {name}", flush=True)
        del m
        torch.cuda.empty_cache()
    for name, rows in sets.items():
        write_jsonl(OUT / f"scored_{name}.jsonl", ({k: v for k, v in r.items() if k not in ("prompt", "response")} for r in rows))
    print("done", flush=True)


# ----------------------------------------------------------------------------------------------- stage: analyze
def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def rm_validation(report: Dict[str, Any]) -> None:
    """Item 1: shared vs separate on the SAME held-out val pairs (per objective, per pair type) + head agreement."""
    def load(run, split="val"):
        return read_jsonl(run / "eval" / f"{split}_rows.jsonl")
    out: Dict[str, Any] = {}
    for split in ("val", "test_representative", "test_balanced"):
        try:
            sh = load(RM_RUNS["shared"], split)
            sr = load(RM_RUNS["separate_reward"], split)
            scst = load(RM_RUNS["separate_cost"], split)
        except FileNotFoundError:
            continue
        res: Dict[str, Any] = {}
        for obj, sep_rows in (("helpfulness", sr), ("harmlessness", scst)):
            shr = {r["pair_id"]: r for r in sh if r["objective"] == obj}
            sep = {r["pair_id"]: r for r in sep_rows if r["objective"] == obj}
            common = sorted(set(shr) & set(sep))
            for pt in ("all", "agree", "conflict"):
                ids = [i for i in common if pt == "all" or shr[i]["pair_type"] == pt]
                if not ids:
                    continue
                a_sh = np.array([shr[i]["margin"] > 0 for i in ids]); a_se = np.array([sep[i]["margin"] > 0 for i in ids])
                l_sh = np.array([shr[i]["rank_loss"] for i in ids]); l_se = np.array([sep[i]["rank_loss"] for i in ids])
                # paired comparison on identical pairs: discordant counts (McNemar) and bootstrap CI of the accuracy difference
                rng = np.random.default_rng(0); idx = rng.integers(0, len(ids), (2000, len(ids)))
                diff = (a_sh[idx].mean(1) - a_se[idx].mean(1))
                res[f"{obj}/{pt}"] = {"n_pairs": len(ids), "acc_shared": float(a_sh.mean()), "acc_separate": float(a_se.mean()),
                                      "acc_diff_shared_minus_separate": float(a_sh.mean() - a_se.mean()),
                                      "acc_diff_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                      "only_shared_correct": int((a_sh & ~a_se).sum()), "only_separate_correct": int((~a_sh & a_se).sum()),
                                      "bt_loss_shared": float(l_sh.mean()), "bt_loss_separate": float(l_se.mean())}
        # head agreement on the same responses: each pair has 2 responses; map reward/cost scores to (safer, harmful)
        def per_response(rows_h, rows_c):
            H = {r["pair_id"]: r for r in rows_h if r["objective"] == "helpfulness"}
            C = {r["pair_id"]: r for r in rows_c if r["objective"] == "harmlessness"}
            R_, C_ = [], []
            for pid in sorted(set(H) & set(C)):
                h, c = H[pid], C[pid]
                # cost row: chosen = safer, rejected = harmful. reward row: chosen = better. agree -> better == safer.
                r_safer, r_harm = (h["score_chosen"], h["score_rejected"]) if h["pair_type"] == "agree" else (h["score_rejected"], h["score_chosen"])
                R_ += [r_safer, r_harm]; C_ += [c["score_chosen"], c["score_rejected"]]
            return np.array(R_), np.array(C_)
        R_sh, C_sh = per_response(sh, sh)
        R_se, C_se = per_response(sr, scst)
        res["head_agreement"] = {"spearman_R_vs_C_shared": spearman(R_sh, C_sh), "spearman_R_vs_C_separate": spearman(R_se, C_se),
                                 "spearman_C_shared_vs_C_separate": spearman(C_sh, C_se), "spearman_R_shared_vs_R_separate": spearman(R_sh, R_se),
                                 "n_responses": int(len(C_sh))}
        out[split] = res
    report["item1_rm_validation"] = out


def dual_audit(report: Dict[str, Any], plt) -> None:
    """Item 2: lambda and C_ema - d over training; verify mu_{k+1} = mu_k + eta * lambda_k * (C_ema,k - d) and its sign."""
    eta = 0.01
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    out = {}
    for j, arm in enumerate(("shared", "separate")):
        recs = read_jsonl(RUNS[arm] / "rollout_batches.jsonl")
        recs = sorted({r["batch_index"]: r for r in recs}.values(), key=lambda r: r["batch_index"])  # dedupe resume overlap
        b = np.array([r["batch_index"] for r in recs]) + 1
        lam_used = np.array([r["lambda_used"] for r in recs]); lam = np.array([r["lambda"] for r in recs])
        viol = np.array([r["violation_ema_minus_d"] for r in recs]); cb = np.array([r["cost_batch_mean"] for r in recs])
        ema = np.array([r["cost_ema"] for r in recs]); d = recs[0]["d"]
        dmu = np.log(lam) - np.log(lam_used)
        expected = eta * lam_used * viol
        ema_re = [cb[0]]
        for c in cb[1:]:
            ema_re.append(0.9 * ema_re[-1] + 0.1 * c)
        chain = np.abs(np.log(lam_used[1:]) - np.log(lam[:-1])) if len(recs) > 1 else np.zeros(1)
        out[arm] = {"n_batches": int(len(recs)), "d": d, "lambda_first": float(lam_used[0]), "lambda_final": float(lam[-1]),
                    "max_abs_update_error": float(np.max(np.abs(dmu - expected))),
                    "sign_agreement_frac": float(np.mean(np.sign(dmu) == np.sign(viol))),
                    "max_abs_ema_recompute_error": float(np.max(np.abs(np.array(ema_re) - ema))),
                    "lambda_used_equals_previous_lambda_max_err": float(chain.max()),
                    "frac_batches_violation_negative": float(np.mean(viol < 0)),
                    "cost_batch_mean_first10": float(cb[:10].mean()), "cost_batch_mean_last50": float(cb[-50:].mean()),
                    "violation_last50": float(viol[-50:].mean())}
        axes[0, j].plot(b, lam); axes[0, j].set_title(f"{arm}: λ (log scale)"); axes[0, j].set_yscale("log")
        axes[1, j].plot(b, cb - d, alpha=0.35, label="batch C − d"); axes[1, j].plot(b, viol, label="EMA C − d")
        axes[1, j].axhline(0, color="k", lw=0.8); axes[1, j].set_title(f"{arm}: cost − d (d = {d:.3f})"); axes[1, j].legend(fontsize=8)
        axes[1, j].set_xlabel("rollout batch")
    fig.tight_layout(); fig.savefig(OUT / "item2_lambda_violation.png", dpi=130); plt.close(fig)
    report["item2_dual"] = out


def scorer_agreement(report: Dict[str, Any], plt) -> None:
    """Item 3: shared cost head vs Beaver cost on the saved response sets (means + per-set Spearman)."""
    order = ["pi0_calibration", "eval_base", "shared_b100", "eval_shared", "separate_b100", "eval_separate", "eval_scalar"]
    out = {}
    for name in order:
        p = OUT / f"scored_{name}.jsonl"
        if not p.exists():
            continue
        rows = read_jsonl(p)
        g = lambda k: np.array([r[k] for r in rows], dtype=float)
        out[name] = {"n": len(rows), "mean_shared_C": float(g("shared_C").mean()), "mean_separate_C": float(g("separate_C").mean()),
                     "mean_beaver_C": float(g("beaver_C").mean()), "mean_shared_R": float(g("shared_R").mean()),
                     "mean_separate_R": float(g("separate_R").mean()), "mean_beaver_R": float(g("beaver_R").mean()),
                     "spearman_sharedC_beaverC": spearman(g("shared_C"), g("beaver_C")),
                     "spearman_separateC_beaverC": spearman(g("separate_C"), g("beaver_C")),
                     "spearman_sharedC_separateC": spearman(g("shared_C"), g("separate_C")),
                     "spearman_sharedR_beaverR": spearman(g("shared_R"), g("beaver_R")),
                     "spearman_sharedR_sharedC": spearman(g("shared_R"), g("shared_C")),
                     "trunc_rate": float(g("truncated").mean()), "len_mean": float(g("n_tokens").mean())}
    report["item3_scorer_agreement"] = out
    # trajectory along the shared policy: pi0 (eval prompts) -> batch 100 (monitoring prompts) -> final (eval prompts)
    traj = [n for n in ("eval_base", "shared_b100", "eval_shared") if n in out]
    if traj:
        x = np.arange(len(traj)); labels = {"eval_base": "π0 (batch 0)", "shared_b100": "batch 100\n(n=32)", "eval_shared": "final (566)"}
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, key, ttl in ((axes[0], "mean_shared_C", "shared cost head"), (axes[1], "mean_separate_C", "separate cost model"),
                             (axes[2], "mean_beaver_C", "Beaver cost")):
            ax.plot(x, [out[n][key] for n in traj], "o-"); ax.set_xticks(x); ax.set_xticklabels([labels[n] for n in traj]); ax.set_title(f"shared policy: {ttl}")
        fig.tight_layout(); fig.savefig(OUT / "item3_cost_means_shared_policy.png", dpi=130); plt.close(fig)
        fig, axes = plt.subplots(1, len(traj), figsize=(4.2 * len(traj), 3.8))
        for ax, n in zip(np.atleast_1d(axes), traj):
            rows = read_jsonl(OUT / f"scored_{n}.jsonl")
            tr = np.array([r["truncated"] for r in rows])
            ax.scatter([r["shared_C"] for r in rows], [r["beaver_C"] for r in rows], s=5, c=np.where(tr, "tab:red", "tab:blue"), alpha=0.5)
            ax.set_title(f"{labels[n].replace(chr(10), ' ')}: ρ={out[n]['spearman_sharedC_beaverC']:.2f}"); ax.set_xlabel("shared cost head"); ax.set_ylabel("Beaver cost")
        fig.tight_layout(); fig.savefig(OUT / "item3_sharedC_vs_beaverC.png", dpi=130); plt.close(fig)


def length_analysis(report: Dict[str, Any], plt) -> None:
    """Item 4: length / truncation over training and their relation to the learned and Beaver cost scores."""
    out: Dict[str, Any] = {"over_training": {}, "per_response": {}}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for j, arm in enumerate(("shared", "separate")):
        recs = sorted({r["batch_index"]: r for r in read_jsonl(RUNS[arm] / "rollout_batches.jsonl")}.values(), key=lambda r: r["batch_index"])
        b = np.array([r["batch_index"] for r in recs]) + 1
        L = np.array([r["completion_len_mean"] for r in recs]); T = np.array([r["truncation_rate"] for r in recs])
        C = np.array([r["cost_batch_mean"] for r in recs])
        out["over_training"][arm] = {"spearman_batchLen_batchC": spearman(L, C), "spearman_batchTrunc_batchC": spearman(T, C),
                                     "len_first10": float(L[:10].mean()), "len_last50": float(L[-50:].mean()),
                                     "trunc_first10": float(T[:10].mean()), "trunc_last50": float(T[-50:].mean())}
        ax = axes[0, j]; ax.plot(b, L, label="mean length"); ax2 = ax.twinx(); ax2.plot(b, T, color="tab:red", alpha=0.6, label="truncation")
        ax.set_title(f"{arm}: rollout length (blue) / truncation (red)"); ax2.set_ylim(0, 1)
        axes[1, j].scatter(L, C, c=b, s=6, cmap="viridis"); axes[1, j].set_xlabel("batch mean length"); axes[1, j].set_ylabel(f"batch mean {arm} cost")
    fig.tight_layout(); fig.savefig(OUT / "item4_length_over_training.png", dpi=130); plt.close(fig)
    names = [n for n in ("pi0_calibration", "eval_base", "shared_b100", "eval_shared", "separate_b100", "eval_separate", "eval_scalar") if (OUT / f"scored_{n}.jsonl").exists()]
    if not names:
        report["item4_length"] = out
        return
    fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 6.2), squeeze=False)
    for j, n in enumerate(names):
        rows = read_jsonl(OUT / f"scored_{n}.jsonl")
        L = np.array([r["n_tokens"] for r in rows], float); T = np.array([r["truncated"] for r in rows], bool)
        res = {"n": len(rows), "trunc_rate": float(T.mean())}
        for col in ("shared_C", "separate_C", "beaver_C"):
            v = np.array([r[col] for r in rows], float)
            res[f"spearman_len_{col}"] = spearman(L, v)
            if T.any() and (~T).any():
                res[f"{col}_truncated_minus_complete"] = float(v[T].mean() - v[~T].mean())
            nt = ~T
            if nt.sum() > 10:
                res[f"spearman_len_{col}_complete_only"] = spearman(L[nt], v[nt])
        out["per_response"][n] = res
        for i, col in enumerate(("shared_C", "beaver_C")):
            axes[i, j].scatter(L, [r[col] for r in rows], s=4, alpha=0.4, c=np.where(T, "tab:red", "tab:blue"))
            axes[i, j].set_title(f"{n}\nρ(len,{col})={res[f'spearman_len_{col}']:.2f}", fontsize=8); axes[i, j].set_xlabel("tokens")
    fig.tight_layout(); fig.savefig(OUT / "item4_length_vs_cost_scores.png", dpi=130); plt.close(fig)
    report["item4_length"] = out


def stage_analyze() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}
    rm_validation(report)
    dual_audit(report, plt)
    if any(OUT.glob("scored_*.jsonl")):
        scorer_agreement(report, plt)
    length_analysis(report, plt)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyze"], required=True)
    a = ap.parse_args()
    stage_score() if a.stage == "score" else stage_analyze()
