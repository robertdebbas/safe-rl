"""Held-out development scoring for the pilots: Beaver evaluators on each run's monitoring samples (the 256 monitoring prompts, which no
pilot or main run ever trains on) at start and end, plus modeled reward, reference KL and truncation from the monitor records.
Works for REBEL runs (monitor/<tag>_samples.jsonl always saved) and GRPO pilots (monitor_save_samples: true). The test set is never
touched. Writes runs/<run>/hpval_dev_eval.json per run.    python scripts/sixarm_hpval_devscore.py runs/sixarm_hpval_*_seed42
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from saferl.eval_score import EVALUATOR_TEMPLATE, PKUScoreModel, load_evaluator_tokenizer, score_pairs  # noqa: E402
from saferl.utils import load_config, read_json, read_jsonl, write_json  # noqa: E402

MODELED = {"scalar_raw_mean": "modeled_scalar", "reward_raw_mean": "modeled_reward", "cost_raw_mean": "modeled_cost",      # REBEL monitor keys
           "scalar_reward_mean": "modeled_scalar", "reward_mean": "modeled_reward", "cost_mean": "modeled_cost"}          # GRPO monitor keys


def main(run_dirs: list[str]) -> None:
    ecfg = load_config(str(ROOT / "configs" / "eval.yaml"), [])
    ev = ecfg["evaluators"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets = {}
    for rd in run_dirs:
        rd = Path(rd)
        for tag in ("start", "final"):
            f = rd / "monitor" / f"{tag}_samples.jsonl"
            if f.exists():
                sets[(rd, tag)] = read_jsonl(f)
    if not sets:
        raise SystemExit("no monitoring samples found (REBEL saves them always; GRPO needs monitor_save_samples: true)")
    scores = {k: {} for k in sets}
    for key, col in (("reward", "beaver_R"), ("cost", "beaver_C")):
        tok = load_evaluator_tokenizer(ev[key]["model"], ev[key].get("revision"))
        m = PKUScoreModel.from_pretrained(ev[key]["model"], device, torch.bfloat16, ev[key].get("revision"))
        for k, rows in sets.items():
            res = score_pairs(m, tok, [r["prompt"] for r in rows], [r["response"] for r in rows], device, ecfg["precision"],
                              ev.get("template", EVALUATOR_TEMPLATE), bool(ev.get("append_eos", True)), int(ev["max_length"]), int(ev["batch_size"]))
            scores[k][col] = res["scores"]
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    for rd in sorted({k[0] for k in sets}):
        out = {"run": rd.name, "development_set": "256 monitoring prompts (never trained on)"}
        for tag in ("start", "final"):
            k = (rd, tag)
            if k not in sets:
                continue
            rows = sets[k]; mon = read_json(rd / "monitor" / f"{tag}.json")
            kl0 = mon["kl_initial"]["kl_k3_token_mean"] if isinstance(mon.get("kl_initial"), dict) else mon.get("kl_mean")
            out[tag] = {"n": len(rows), "beaver_helpfulness": statistics.fmean(scores[k]["beaver_R"]), "beaver_harmfulness": statistics.fmean(scores[k]["beaver_C"]),
                        "truncation_rate": mon["truncation_rate"], "completion_len_mean": mon["completion_len_mean"], "kl_initial_token": kl0,
                        **{v: mon[kk] for kk, v in MODELED.items() if kk in mon}, "style_therapy_terms_per_100w": mon.get("style_therapy_terms_per_100w")}
        if "start" in out and "final" in out:
            out["delta"] = {k: out["final"][k] - out["start"][k] for k in ("beaver_helpfulness", "beaver_harmfulness", "truncation_rate", "completion_len_mean")}
            head = "modeled_scalar" if "modeled_scalar" in out["final"] else "modeled_reward"
            out["delta"]["modeled_reward"] = out["final"][head] - out["start"][head]
        write_json(rd / "hpval_dev_eval.json", out)
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or [str(p) for p in sorted(Path("runs").glob("sixarm_hpval_*_seed42"))])
