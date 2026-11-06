"""Within-pair length-bias diagnostic of the learned reward models on the 5,000 AGREEMENT training pairs.

For each pair: dL = L(y+) - L(y-) (Qwen2.5 tokens of the response text) and dR = R(y+) - R(y-) under
  - the shared model's reward head  (runs/rm_shared_qwen2.5-1.5b_seed42/checkpoints/best)
  - the separate reward model       (runs/rm_separate_reward_qwen2.5-1.5b_seed42/checkpoints/best)
y+ = helpfulness-preferred response (= safer response, since these are agreement pairs).
Scoring uses the exact RM training encoding (encode_rows + compute_scores). Read-only; writes runs/diagnose_shared/.

  --stage score    (GPU)  -> runs/diagnose_shared/length_bias_train_agreement.jsonl
  --stage analyze  (CPU)  -> correlations JSON + scatter/fit plot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "diagnose_shared"
SCORES = OUT / "length_bias_train_agreement.jsonl"
MODELS = {"shared_reward_head": ROOT / "runs" / "rm_shared_qwen2.5-1.5b_seed42" / "checkpoints" / "best",
          "separate_reward_model": ROOT / "runs" / "rm_separate_reward_qwen2.5-1.5b_seed42" / "checkpoints" / "best"}


def stage_score() -> None:
    import torch
    from saferl.modeling import encode_rows, load_tokenizer
    from saferl.rm_data import load_rm_rows
    from saferl.rm_models import ScoreModel
    from saferl.train_dpo import resolve_lengths
    from saferl.train_rm import compute_scores
    from saferl.utils import load_config

    cfg = load_config(str(ROOT / "configs" / "rm_common.yaml"), [])
    lengths = resolve_lengths(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(cfg["model"])
    rows = [r for r in load_rm_rows(cfg["artifacts_dir"], "train", ["helpfulness"]) if r["pair_type"] == "agreement"]
    assert len(rows) == 5000, len(rows)
    enc = encode_rows(tok, rows, lengths["max_prompt_length"], lengths["max_length"], "error")
    out = {r["row_id"]: {"pair_id": r["pair_id"],
                         "len_pos": len(tok(r["chosen"], add_special_tokens=False)["input_ids"]),
                         "len_neg": len(tok(r["rejected"], add_special_tokens=False)["input_ids"])} for r in rows}
    for name, path in MODELS.items():
        m = ScoreModel.load(str(path), device, torch.float32)
        sc = compute_scores(m, enc, rows, tok.pad_token_id, int(cfg["eval_batch_size"]), device, cfg["precision"],
                            {"helpfulness": "reward"}, name)
        for rid, s in sc.items():
            out[rid][f"{name}_pos"] = s["chosen"]; out[rid][f"{name}_neg"] = s["rejected"]
        del m
        torch.cuda.empty_cache()
        print(f"scored {name}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(SCORES, "w") as f:
        for v in out.values():
            f.write(json.dumps(v) + "\n")
    print("done", flush=True)


def stage_analyze() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    rows = [json.loads(l) for l in open(SCORES)]
    dL = np.array([r["len_pos"] - r["len_neg"] for r in rows], float)
    res = {"n_pairs": len(rows)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    for ax, name in zip(axes, MODELS):
        dR = np.array([r[f"{name}_pos"] - r[f"{name}_neg"] for r in rows], float)
        slope, icpt = np.polyfit(dL, dR, 1)
        res[name] = {"pearson": float(pearsonr(dL, dR)[0]), "spearman": float(spearmanr(dL, dR).correlation),
                     "ols_slope_per_100_tokens": float(slope * 100), "ols_intercept": float(icpt),
                     "pair_accuracy": float(np.mean(dR > 0)), "mean_dR": float(dR.mean())}
        ax.scatter(dL, dR, s=3, alpha=0.25)
        xs = np.linspace(dL.min(), dL.max(), 50); ax.plot(xs, slope * xs + icpt, color="tab:red")
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{name}: Pearson {res[name]['pearson']:.2f}, Spearman {res[name]['spearman']:.2f}")
        ax.set_xlabel("ΔL = L(y+) − L(y−)  [tokens]"); ax.set_ylabel("ΔR = R(y+) − R(y−)")
    fig.tight_layout(); fig.savefig(OUT / "length_bias_train_agreement.png", dpi=130); plt.close(fig)
    (OUT / "length_bias_train_agreement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyze"], required=True)
    stage_score() if ap.parse_args().stage == "score" else stage_analyze()
