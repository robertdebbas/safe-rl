"""Evaluate a trained DPO checkpoint (default: the run's best-by-validation checkpoint) on
val / test_representative / test_balanced with subgroup metrics and the conflict diagnostic.

The reference model is the base checkpoint named in the config; its log-probs use the same
cache as training (shared across seeds)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .metrics import render_metrics_markdown, row_records, summarize
from .modeling import compute_logprobs, encode_rows, get_or_compute_ref_logprobs, load_model, load_tokenizer, ref_cache_key
from .prepare import load_split_rows
from .train_dpo import resolve_lengths
from .utils import get_logger, load_config, pick_device, read_json, write_json, write_jsonl

log = get_logger("saferl.eval")


def evaluate_checkpoint(cfg: Dict[str, Any], run_dir: Path, checkpoint: Optional[str] = None,
                        splits: Optional[List[str]] = None) -> Dict[str, Any]:
    device = pick_device(cfg.get("device", "auto"))
    precision = cfg["precision"]
    lengths = resolve_lengths(cfg)
    beta, ls = float(cfg["beta"]), float(cfg.get("label_smoothing", 0.0))
    splits = splits or cfg.get("eval_splits", ["val", "test_representative", "test_balanced"])
    eval_bs = int(cfg["eval_batch_size"])
    if checkpoint is None:
        checkpoint = str(run_dir / "checkpoints" / "best")
    ckpt_path = Path(checkpoint).resolve()
    state = read_json(ckpt_path / "trainer_state.json") if (ckpt_path / "trainer_state.json").exists() else {}
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("evaluating %s on %s (device=%s)", ckpt_path, splits, device)

    tokenizer = load_tokenizer(cfg["model"])
    hashes = read_json(Path(cfg["artifacts_dir"]) / "splits" / "manifest_hashes.json")
    rows = {s: load_split_rows(cfg["artifacts_dir"], s) for s in splits}
    enc = {s: encode_rows(tokenizer, rows[s], lengths["max_prompt_length"], lengths["max_length"], "error") for s in splits}
    pad_id = tokenizer.pad_token_id

    # reference log-probs (base model) --------------------------------------------------------
    ref: Dict[str, Dict[str, Dict[str, float]]] = {}
    missing = []
    cache_dir = cfg.get("ref_cache_dir")
    keys = {s: ref_cache_key(cfg["model"], hashes[s]["dpo_rows_sha256"], lengths["max_prompt_length"], lengths["max_length"], device, precision) for s in splits}
    for s in splits:
        p = Path(cache_dir) / f"{s}.{keys[s]}.json" if cache_dir else None
        if p is not None and p.exists():
            ref[s] = read_json(p)
        else:
            missing.append(s)
    if missing:
        base = load_model(cfg["model"], device, torch.float32)
        for s in missing:
            ref[s], _ = get_or_compute_ref_logprobs(cache_dir, s, keys[s], lambda s=s: compute_logprobs(base, enc[s], pad_id, eval_bs, device, precision, f"ref:{s}"))
        del base
        if device == "cuda":
            torch.cuda.empty_cache()

    # policy ----------------------------------------------------------------------------------
    policy = load_model(str(ckpt_path), device, torch.float32)
    results: Dict[str, Any] = {"checkpoint": str(ckpt_path), "checkpoint_state": state, "beta": beta, "splits": {}}
    md = [f"# Evaluation: {run_dir.name}\n", f"checkpoint: `{ckpt_path}` (step {state.get('step')}, combined val loss at save {state.get('combined_val_loss')})\n"]
    for s in splits:
        t0 = time.time()
        pol = compute_logprobs(policy, enc[s], pad_id, eval_bs, device, precision, f"policy:{s}")
        recs = row_records(rows[s], pol, ref[s], beta, ls)
        m = summarize(recs, beta)
        m["seconds"] = time.time() - t0
        results["splits"][s] = m
        write_json(out_dir / f"{s}.json", m)
        write_jsonl(out_dir / f"{s}_rows.jsonl", recs)
        md.append(render_metrics_markdown(m, s))
        log.info("%s: combined=%.4f acc=%.3f margin=%.4f (%.0fs)", s, m["combined_loss"], m["groups"]["overall"]["pref_acc"],
                 m["groups"]["overall"]["mean_margin"], m["seconds"])
    write_json(out_dir / "eval_summary.json", results)
    (out_dir / "eval_summary.md").write_text("\n".join(md), encoding="utf-8")
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default=None, help="defaults to <run_dir>/checkpoints/best")
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    evaluate_checkpoint(cfg, Path(args.run_dir), args.checkpoint, args.splits)


if __name__ == "__main__":
    main()
