#!/usr/bin/env python3
"""GPU pilot / final sanity check of the evaluation pipeline (no scientific settings are changed).

Runs, on a small prompt subset of the real manifest with Base Qwen and the real PKU Beaver evaluators:
  1. evaluator loading: reproduces the model-card "hello" example (end score and first per-token score),
     checks BOS / appended-EOS / last-token extraction and padding invariance;
  2. peak GPU memory for generation, reward scoring, cost scoring (torch.cuda peak stats);
  3. generation throughput, completion-length distribution, fraction at the 512-token cap;
  4. scoring throughput at the configured batch size, finiteness and spread of both score distributions;
  5. reproducibility: regenerate the first batch with the same per-prompt seeds; sharded (2) vs unsharded (1)
     generation + scoring on the subset must be byte-identical;
  6. artifacts: responses saved before scores, metadata fields present (evaluator revisions, checkpoint
     fingerprint, manifest hash, generation settings, per-response scores).
Writes <out>/pilot_report.{json,md}.

    python scripts/eval_pilot.py --config configs/eval.yaml --n-prompts 40          # cluster, one GPU
    python scripts/eval_pilot.py --config configs/eval_smoke.yaml --n-prompts 4 --out runs/eval_pilot_smoke   # local smoke
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import saferl  # noqa: E402,F401
from saferl.eval_common import EVALUATOR_TEMPLATE, load_eval_prompts, prompt_seed  # noqa: E402
from saferl.eval_generate import generate_shard, sample_with_prompt_seeds  # noqa: E402
from saferl.eval_merge import merge_shards  # noqa: E402
from saferl.eval_score import PKUScoreModel, encode_for_evaluator, load_evaluator_tokenizer, score_shard  # noqa: E402
from saferl.modeling import autocast_ctx, load_model, load_tokenizer  # noqa: E402
from saferl.utils import get_logger, load_config, pick_device, read_json, read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402

log = get_logger("saferl.eval_pilot")

CARD_EXAMPLE = {"prompt": "hello", "response": "Hello! How can I help you today?",
                "reward": {"end_score": 0.3789, "first_token_score": -7.2812}, "cost": {"end_score": -7.6562, "first_token_score": -2.7656}}


def mem_reset(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def mem_peak(device: str) -> dict:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return {"peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                "device_total_gb": total / 2**30}
    if device == "mps":
        return {"current_allocated_gb": torch.mps.current_allocated_memory() / 2**30, "driver_gb": torch.mps.driver_allocated_memory() / 2**30,
                "note": "MPS has no peak counter"}
    return {}


def pct(a) -> dict:
    a = np.asarray(a, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "p5": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def check_evaluator(cfg: dict, key: str, device: str, report: dict) -> None:
    ev = cfg["evaluators"][key]
    repo = ev["model"]
    tok = load_evaluator_tokenizer(repo, ev.get("revision"))
    mem_reset(device)
    t0 = time.time()
    model = PKUScoreModel.from_pretrained(repo, device, torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32, ev.get("revision"))
    load_s = time.time() - t0
    out = {"model": repo, "revision": ev.get("revision"), "score_type": model.score_type, "load_seconds": load_s,
           "n_parameters": sum(p.numel() for p in model.parameters()), "head_bias": model.score_head.bias is not None}
    # --- model-card example: exactly as documented (no appended EOS) -> end score + first per-token score
    text = EVALUATOR_TEMPLATE.format(prompt=CARD_EXAMPLE["prompt"], response=CARD_EXAMPLE["response"])
    ids = tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    attn = torch.ones_like(ids)
    with torch.no_grad(), autocast_ctx(device, cfg["precision"]):
        h = model.backbone(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
        per_tok = model.score_head(h).squeeze(-1).float()[0]
    card = CARD_EXAMPLE[key] if model.score_type in ("reward", "cost") and "unified" in repo else None
    out["card_example"] = {"template_text": text, "n_tokens": int(ids.shape[1]), "bos_first": int(ids[0, 0]) == tok.bos_token_id,
                           "end_score_no_eos": float(per_tok[-1]), "first_token_score": float(per_tok[0]),
                           "documented_end_score": card["end_score"] if card else None, "documented_first_token_score": card["first_token_score"] if card else None,
                           "abs_diff_end": abs(float(per_tok[-1]) - card["end_score"]) if card else None,
                           "abs_diff_first": abs(float(per_tok[0]) - card["first_token_score"]) if card else None}
    # --- our convention: appended EOS id, score at the last non-pad token; padding invariance
    enc1 = encode_for_evaluator(tok, [CARD_EXAMPLE["prompt"]], [CARD_EXAMPLE["response"]], cfg["evaluators"].get("template", EVALUATOR_TEMPLATE),
                                bool(cfg["evaluators"].get("append_eos", True)), int(cfg["evaluators"]["max_length"]))
    ids1 = enc1["input_ids"][0, : enc1["lengths"][0]].tolist()
    enc2 = encode_for_evaluator(tok, [CARD_EXAMPLE["prompt"], "a much longer prompt " * 20], [CARD_EXAMPLE["response"], "x"],
                                cfg["evaluators"].get("template", EVALUATOR_TEMPLATE), bool(cfg["evaluators"].get("append_eos", True)), int(cfg["evaluators"]["max_length"]))
    with torch.no_grad(), autocast_ctx(device, cfg["precision"]):
        s1 = float(model(enc1["input_ids"].to(device), enc1["attention_mask"].to(device))[0])
        s2 = float(model(enc2["input_ids"].to(device), enc2["attention_mask"].to(device))[0])
    out["formatting"] = {"bos_first": ids1[0] == tok.bos_token_id, "eos_last": ids1[-1] == tok.eos_token_id,
                         "decoded": tok.decode(ids1, skip_special_tokens=False), "score_with_eos": s1, "score_with_eos_padded_batch": s2,
                         "padding_invariant": abs(s1 - s2) < 0.05, "last_token_index_rule": "attention_mask.sum(1) - 1"}
    out["memory_after_load_and_probe"] = mem_peak(device)
    report["evaluators"][key] = out
    log.info("%s: card end score %.4f (documented %s), with-EOS score %.4f, padded %.4f, load %.0fs", key, float(per_tok[-1]),
             card["end_score"] if card else "n/a", s1, s2, load_s)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--method", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-prompts", type=int, default=40, help="pilot subset = first N manifest prompts (multiple of the generation batch size)")
    ap.add_argument("--out", default="runs/eval_pilot")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    device = pick_device(cfg.get("device", "auto"))
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    report: dict = {"config": args.config, "method": args.method, "seed": args.seed, "device": device, "n_prompts": args.n_prompts,
                    "torch": torch.__version__, "evaluators": {}}
    if device.startswith("cuda"):
        report["gpu"] = {"name": torch.cuda.get_device_name(0), "total_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}

    # ---------------------------------------------------------------- pilot manifest subset (first N prompts)
    prompts = load_eval_prompts(cfg["artifacts_dir"])
    bs = int(cfg["generation"]["batch_size"])
    n = min(args.n_prompts, len(prompts))
    if n % bs != 0:
        raise SystemExit(f"--n-prompts must be a multiple of generation.batch_size={bs}")
    sub_art = out / "artifacts"
    (sub_art / "eval").mkdir(parents=True)
    write_jsonl(sub_art / "eval" / "eval_prompts.jsonl", prompts[:n])
    full_info = read_json(Path(cfg["artifacts_dir"]) / "eval" / "manifest_info.json")
    write_json(sub_art / "eval" / "manifest_info.json", {**full_info, "pilot_subset_of": full_info["sha256"], "n_prompts": n,
                                                           "sha256": sha256_file(sub_art / "eval" / "eval_prompts.jsonl")})
    pcfg = {**cfg, "artifacts_dir": str(sub_art), "out_dir": str(out / "unsharded")}
    report["manifest"] = {"full_sha256": full_info["sha256"], "full_n_prompts": full_info["n_prompts"], "pilot_n_prompts": n}

    # ---------------------------------------------------------------- 1. evaluators
    for key in ("reward", "cost"):
        check_evaluator(cfg, key, device, report)

    # ---------------------------------------------------------------- 2+3. generation (unsharded)
    mem_reset(device)
    t0 = time.time()
    gsum = generate_shard(pcfg, args.method, args.seed, 1, 0, overwrite=True)
    gen_s = time.time() - t0
    gen_mem = mem_peak(device)
    resp_file = Path(gsum["file"])
    resp_saved_at = resp_file.stat().st_mtime
    recs = read_jsonl(resp_file)
    lens = [r["n_completion_tokens"] for r in recs]
    report["generation"] = {"summary_file": str(resp_file.with_suffix(".summary.json")), "wallclock_s": gen_s, "seconds_per_prompt": gen_s / n,
                            "seconds_per_batch": gen_s / (n / bs), "batch_size": bs, "completion_tokens": pct(lens),
                            "fraction_at_cap": float(np.mean([r["truncated"] for r in recs])), "max_new_tokens": cfg["generation"]["max_new_tokens"],
                            "tokens_per_second": float(sum(lens) / gen_s), "peak_memory": gen_mem,
                            "checkpoint_fingerprint": gsum.get("checkpoint_fingerprint")}
    log.info("generation: %.1fs for %d prompts (%.2fs/prompt), len p50 %.0f p95 %.0f, at cap %.3f, mem %s", gen_s, n, gen_s / n,
             report["generation"]["completion_tokens"]["p50"], report["generation"]["completion_tokens"]["p95"], report["generation"]["fraction_at_cap"], gen_mem)

    # ---------------------------------------------------------------- 4. scoring (unsharded), per-evaluator peaks via monkeypatched hook
    mem_reset(device)
    t0 = time.time()
    ssum = score_shard(pcfg, args.method, args.seed, 1, 0, overwrite=True)
    score_s = time.time() - t0
    score_file = Path(ssum["file"])
    scores = read_jsonl(score_file)
    r = np.array([x["reward_score"] for x in scores]); c = np.array([x["cost_score"] for x in scores])
    report["scoring"] = {"wallclock_both_evaluators_s": score_s, "responses_per_second_both": n / score_s, "batch_size": cfg["evaluators"]["batch_size"],
                         "peak_memory_both_sequential": mem_peak(device), "reward": pct(r), "cost": pct(c),
                         "all_finite": bool(np.isfinite(r).all() and np.isfinite(c).all()),
                         "reward_distinct_values": int(len(np.unique(np.round(r, 4)))), "cost_distinct_values": int(len(np.unique(np.round(c, 4)))),
                         "evaluator_input_truncated": int(sum(x["evaluator_input_truncated"] for x in scores)),
                         "responses_saved_before_scores": resp_saved_at <= score_file.stat().st_mtime}
    log.info("scoring: %.1fs for %d responses (both evaluators); reward mean %.3f std %.3f; cost mean %.3f std %.3f; finite %s", score_s, n, r.mean(), r.std(),
             c.mean(), c.std(), report["scoring"]["all_finite"])

    # ---------------------------------------------------------------- 5a. reproducibility: regenerate the first batch
    tok = load_tokenizer(gsum["checkpoint"])
    model = load_model(gsum["checkpoint"], device, torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32, attn_implementation=cfg.get("attn_implementation"))
    first = prompts[:bs]
    seeds = [prompt_seed(int(cfg["eval_seed"]), p["prompt_hash"]) for p in first]
    g = cfg["generation"]
    again = sample_with_prompt_seeds(model, tok, [p["prompt"] for p in first], seeds, g["max_prompt_length"], g["max_new_tokens"], device, cfg["precision"],
                                     g["temperature"], g["top_p"])
    orig = {x["prompt_hash"]: x["completion_ids"] for x in recs}
    same = [again[i]["completion_ids"] == orig[p["prompt_hash"]] for i, p in enumerate(first)]
    report["reproducibility"] = {"regenerated_first_batch_identical": all(same), "n_identical": int(sum(same)), "n_checked": len(same)}
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 5b. sharded (2) vs unsharded
    if n // bs >= 2:
        scfg = {**pcfg, "out_dir": str(out / "sharded")}
        for i in range(2):
            generate_shard(scfg, args.method, args.seed, 2, i, overwrite=True)
            score_shard(scfg, args.method, args.seed, 2, i, overwrite=True)
        hashes = [p["prompt_hash"] for p in prompts[:n]]
        d1 = Path(pcfg["out_dir"]) / args.method / f"seed{args.seed}"
        d2 = Path(scfg["out_dir"]) / args.method / f"seed{args.seed}"
        r1 = merge_shards(d1, "responses", hashes)["rows"]; r2 = merge_shards(d2, "responses", hashes)["rows"]
        s1 = merge_shards(d1, "scores", hashes)["rows"]; s2 = merge_shards(d2, "scores", hashes)["rows"]
        same_resp = all(a["completion_ids"] == b["completion_ids"] for a, b in zip(r1, r2))
        same_scores = all(a["reward_score"] == b["reward_score"] and a["cost_score"] == b["cost_score"] for a, b in zip(s1, s2))
        report["reproducibility"]["sharded_vs_unsharded_responses_identical"] = same_resp
        report["reproducibility"]["sharded_vs_unsharded_scores_identical"] = same_scores
        report["reproducibility"]["max_abs_score_diff"] = float(max(max(abs(a["reward_score"] - b["reward_score"]), abs(a["cost_score"] - b["cost_score"])) for a, b in zip(s1, s2)))

    # ---------------------------------------------------------------- 6. artifacts
    gmeta = read_json(resp_file.with_suffix(".summary.json")); smeta = read_json(score_file.with_suffix(".summary.json"))
    report["artifacts"] = {
        "responses_file": str(resp_file), "scores_file": str(score_file),
        "generation_summary_has": {k: (k in gmeta) for k in ("checkpoint", "checkpoint_fingerprint", "manifest_sha256", "generation", "eval_seed")},
        "scoring_summary_has": {k: (k in smeta) for k in ("evaluators", "template", "append_eos", "max_length", "batch_size")},
        "evaluator_revisions_recorded": {k: smeta["evaluators"][k].get("revision") for k in ("reward", "cost")},
        "per_response_fields": sorted(scores[0].keys()),
    }
    ok_80 = None
    if device.startswith("cuda"):
        peaks = [report["generation"]["peak_memory"]["peak_reserved_gb"], report["scoring"]["peak_memory_both_sequential"]["peak_reserved_gb"]] + \
                [report["evaluators"][k]["memory_after_load_and_probe"]["peak_reserved_gb"] for k in ("reward", "cost")]
        ok_80 = max(peaks) < 60.0
        report["one_80gb_gpu_sufficient"] = {"max_peak_reserved_gb": max(peaks), "comfortable_below_60gb": ok_80}
    write_json(out / "pilot_report.json", report)
    md = [f"# Evaluation pilot ({args.method}, {n} prompts, device {device})\n", "```json", json.dumps({k: v for k, v in report.items() if k != "config"}, indent=1), "```"]
    (out / "pilot_report.md").write_text("\n".join(md), encoding="utf-8")
    log.info("pilot report written to %s", out / "pilot_report.json")


if __name__ == "__main__":
    main()
