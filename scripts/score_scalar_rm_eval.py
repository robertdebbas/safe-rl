"""Add the scalar RM's score (column ``scalar_S``) to runs/diagnose_shared/scored_eval_*.jsonl, in place.
Reads the existing eval responses (text) and the frozen scalar RM (RM seed 42, best-by-val) — scoring only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from saferl.grpo_scoring import FrozenScorers  # noqa: E402
from saferl.modeling import load_tokenizer  # noqa: E402
from saferl.utils import load_config  # noqa: E402

OUT = ROOT / "runs" / "diagnose_shared"
SETS = {"eval_base": ("base", 0), "eval_scalar": ("grpo_scalar", 42), "eval_shared": ("grpo_shared", 42), "eval_separate": ("grpo_separate", 42)}

cfg = load_config(str(ROOT / "configs" / "grpo_common.yaml"), [])
gen = cfg["generation"]
tok = load_tokenizer(cfg["policy_model"])
sc = FrozenScorers("scalar", {"scalar": str(ROOT / "runs" / "rm_scalar_qwen2.5-1.5b_seed42" / "checkpoints" / "best")},
                   "cuda", cfg["precision"], int(cfg.get("scorer_batch_size", 16)), gen["max_prompt_length"],
                   gen["max_prompt_length"] + gen["max_completion_length"])
for name, (method, seed) in SETS.items():
    text = {r["prompt_hash"]: r for r in map(json.loads, open(ROOT / "runs" / "eval" / method / f"seed{seed}" / "responses.jsonl"))}
    rows = [json.loads(l) for l in open(OUT / f"scored_{name}.jsonl")]
    s = sc.score(tok, [text[r["prompt_hash"]]["prompt"] for r in rows], [text[r["prompt_hash"]]["response"] for r in rows])["scalar"]
    for r, v in zip(rows, s.tolist()):
        r["scalar_S"] = float(v)
    with open(OUT / f"scored_{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"scalar scored {name}", flush=True)
print("done", flush=True)

# ---- Beaver helpfulness/harmfulness on the 2,000 val preference pairs (4,000 responses), for part-2 agreement with Beaver
del sc
torch.cuda.empty_cache()
from saferl.eval_score import EVALUATOR_TEMPLATE, PKUScoreModel, load_evaluator_tokenizer, score_pairs  # noqa: E402
ecfg = load_config(str(ROOT / "configs" / "eval.yaml"), [])
ev = ecfg["evaluators"]
pairs = [json.loads(l) for l in open(ROOT / "artifacts" / "splits" / "val.pairs.jsonl")]
prompts = [p["prompt"] for p in pairs for _ in (0, 1)]
responses = [p[f"response_{i}"] for p in pairs for i in (0, 1)]
res = {}
for key in ("reward", "cost"):
    etok = load_evaluator_tokenizer(ev[key]["model"], ev[key].get("revision"))
    m = PKUScoreModel.from_pretrained(ev[key]["model"], "cuda", torch.bfloat16, ev[key].get("revision"))
    res[key] = score_pairs(m, etok, prompts, responses, "cuda", ecfg["precision"], ev.get("template", EVALUATOR_TEMPLATE),
                           bool(ev.get("append_eos", True)), int(ev["max_length"]), int(ev["batch_size"]))["scores"]
    del m
    torch.cuda.empty_cache()
    print(f"beaver {key} scored val pairs", flush=True)
with open(OUT / "beaver_val_pairs.jsonl", "w") as f:
    for j, p in enumerate(pairs):
        f.write(json.dumps({"pair_id": p["pair_id"], "pair_type": p["pair_type"], "better_response_id": p["better_response_id"],
                            "safer_response_id": p["safer_response_id"], "beaver_R": [res["reward"][2 * j], res["reward"][2 * j + 1]],
                            "beaver_C": [res["cost"][2 * j], res["cost"][2 * j + 1]]}) + "\n")
print("done val pairs", flush=True)
