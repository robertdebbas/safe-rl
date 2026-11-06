"""Evaluate a trained score model (default: the run's best-by-validation checkpoint) on
val / test_representative / test_balanced, per objective x pair type, pair- and prompt-averaged."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .modeling import encode_rows, load_tokenizer
from .rm_data import load_rm_rows
from .rm_metrics import render_metrics_markdown, row_records, summarize
from .rm_models import ScoreModel
from .train_dpo import resolve_lengths
from .train_rm import arch_spec, compute_scores
from .utils import get_logger, load_config, pick_device, read_json, write_json, write_jsonl

log = get_logger("saferl.eval_rm")


def evaluate_checkpoint(cfg: Dict[str, Any], run_dir: Path, checkpoint: Optional[str] = None,
                        splits: Optional[List[str]] = None) -> Dict[str, Any]:
    spec = arch_spec(cfg)
    device = pick_device(cfg.get("device", "auto"))
    precision = cfg["precision"]
    lengths = resolve_lengths(cfg)
    splits = splits or cfg.get("eval_splits", ["val", "test_representative", "test_balanced"])
    eval_bs = int(cfg["eval_batch_size"])
    ckpt_path = Path(checkpoint or (run_dir / "checkpoints" / "best")).resolve()
    state = read_json(ckpt_path / "trainer_state.json") if (ckpt_path / "trainer_state.json").exists() else {}
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("evaluating %s (arch=%s) on %s", ckpt_path, cfg["arch"], splits)

    tokenizer = load_tokenizer(cfg["model"])
    model = ScoreModel.load(str(ckpt_path), device, torch.float32)
    results: Dict[str, Any] = {"checkpoint": str(ckpt_path), "checkpoint_state": state, "arch": cfg["arch"],
                               "heads": spec["heads"], "cost_pointwise_classification": spec["cost_pointwise_classification"],
                               "parameters": model.num_parameters(), "splits": {}}
    md = [f"# Evaluation: {run_dir.name} (arch={cfg['arch']})\n",
          f"checkpoint: `{ckpt_path}` (step {state.get('step')}, combined val loss at save {state.get('combined_val_loss')})\n"]
    for s in splits:
        t0 = time.time()
        rows = load_rm_rows(cfg["artifacts_dir"], s, spec["objectives"])
        enc = encode_rows(tokenizer, rows, lengths["max_prompt_length"], lengths["max_length"], "error")
        sc = compute_scores(model, enc, rows, tokenizer.pad_token_id, eval_bs, device, precision, spec["head_for"], f"{s}")
        recs = row_records(rows, sc, spec["loss_for"], spec["cost_pointwise_classification"])
        m = summarize(recs)
        m["seconds"] = time.time() - t0
        results["splits"][s] = m
        write_json(out_dir / f"{s}.json", m)
        write_jsonl(out_dir / f"{s}_rows.jsonl", recs)
        md.append(render_metrics_markdown(m, s))
        log.info("%s: combined=%.4f acc_tie_adj=%.3f (%.0fs)", s, m["combined_loss"], m["groups"]["overall"]["acc_tie_adjusted"], m["seconds"])
    write_json(out_dir / "eval_summary.json", results)
    (out_dir / "eval_summary.md").write_text("\n".join(md), encoding="utf-8")
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    evaluate_checkpoint(load_config(args.config, args.set), Path(args.run_dir), args.checkpoint, args.splits)


if __name__ == "__main__":
    main()
