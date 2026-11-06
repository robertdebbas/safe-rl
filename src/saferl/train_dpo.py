"""Vanilla sigmoid DPO trainer (full-parameter fine-tuning, frozen reference = initial checkpoint).

Single-process implementation (world_size must be 1; see README for the multi-GPU status).
fp32 master weights + bf16 autocast; reference log-probs are computed once with the model at
initialization (policy == reference at step 0) and cached on disk, shared across training seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .data import OBJECTIVES, PAIR_TYPES
from .dpo_loss import dpo_loss
from .formatting import assert_no_objective_leakage, collate_dpo
from .metrics import render_metrics_markdown, row_records, summarize
from .modeling import (compute_logprobs, encode_rows, forward_logprobs, get_or_compute_ref_logprobs, load_model,
                       load_tokenizer, ref_cache_key, save_checkpoint)
from .prepare import load_length_limits, load_split_rows
from .utils import get_logger, load_config, pick_device, read_json, seed_everything, world_size, write_json

log = get_logger("saferl.train")


# --------------------------------------------------------------------------- helpers

def free_cache(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def resolve_batch_config(cfg: Dict[str, Any]) -> Dict[str, int]:
    ws = world_size()
    if ws != 1:
        raise RuntimeError(f"WORLD_SIZE={ws}: this trainer is single-process only. Launch one process on one GPU.")
    gbs, pdbs = int(cfg["global_batch_size"]), int(cfg["per_device_batch_size"])
    if gbs % (pdbs * ws) != 0:
        raise RuntimeError(
            f"global_batch_size={gbs} is not divisible by per_device_batch_size*world_size={pdbs}*{ws}. "
            "Refusing to silently change the global batch.")
    accum = gbs // (pdbs * ws)
    assert pdbs * ws * accum == gbs
    return {"world_size": ws, "per_device_batch_size": pdbs, "gradient_accumulation_steps": accum, "global_batch_size": gbs}


def resolve_lengths(cfg: Dict[str, Any]) -> Dict[str, int]:
    if cfg.get("max_prompt_length") and cfg.get("max_length"):
        return {"max_prompt_length": int(cfg["max_prompt_length"]), "max_length": int(cfg["max_length"]), "source": "config"}
    lim = load_length_limits(cfg["artifacts_dir"])
    return {"max_prompt_length": int(lim["max_prompt_length"]), "max_length": int(lim["max_length"]), "source": "artifacts/length_limits.json"}


def build_scheduler(optimizer, cfg: Dict[str, Any], total_steps: int):
    from transformers import get_scheduler

    warmup = int(round(float(cfg["warmup_ratio"]) * total_steps))
    return get_scheduler(cfg["lr_scheduler"], optimizer=optimizer, num_warmup_steps=warmup, num_training_steps=total_steps), warmup


def epoch_order(n: int, seed: int, epoch: int) -> List[int]:
    """Independent shuffle of all DPO rows each epoch (H and S rows of a pair are not paired)."""
    g = torch.Generator().manual_seed(int(seed) * 1000 + int(epoch))
    return torch.randperm(n, generator=g).tolist()


class CheckpointManager:
    """Keeps best (by combined validation loss), last, and epoch-end checkpoints; deletes the rest."""

    def __init__(self, root: Path, keep: List[str], dtype: str, save_fn=None):
        self.root, self.keep, self.dtype = root, set(keep), dtype
        self.save_fn = save_fn  # (model, tokenizer, path, dtype, state) -> None; defaults to modeling.save_checkpoint
        self.best_step: Optional[int] = None
        self.best_loss = float("inf")
        self.epoch_end_steps: set = set()
        self.last_step: Optional[int] = None
        self.saved: Dict[int, Path] = {}

    def path(self, step: int) -> Path:
        return self.root / f"step_{step:06d}"

    def save(self, model, tokenizer, step: int, epoch: float, metrics: Dict[str, Any], is_epoch_end: bool) -> bool:
        loss = metrics["combined_loss"]
        is_best = loss < self.best_loss
        state = {"step": step, "epoch": epoch, "combined_val_loss": loss, "is_epoch_end": is_epoch_end, "is_best_at_save": is_best}
        (self.save_fn or save_checkpoint)(model, tokenizer, self.path(step), self.dtype, state)
        self.saved[step] = self.path(step)
        if is_best:
            self.best_loss, self.best_step = loss, step
        if is_epoch_end:
            self.epoch_end_steps.add(step)
        self.last_step = step
        self._prune()
        return is_best

    def _protected(self) -> set:
        p = set()
        if "best" in self.keep and self.best_step is not None:
            p.add(self.best_step)
        if "last" in self.keep and self.last_step is not None:
            p.add(self.last_step)
        if "epoch_end" in self.keep:
            p |= self.epoch_end_steps
        if "all" in self.keep:
            p |= set(self.saved)
        return p

    def _prune(self) -> None:
        keep = self._protected()
        for step in list(self.saved):
            if step not in keep:
                shutil.rmtree(self.saved[step], ignore_errors=True)
                del self.saved[step]

    def finalize(self) -> Dict[str, Any]:
        """``best`` (diagnostic: lowest combined validation loss) and ``final`` (the last saved step, i.e. the
        end-of-training checkpoint used for the PRIMARY comparison) are symlinks to step directories."""
        best = self.path(self.best_step) if self.best_step is not None else None
        final = self.path(self.last_step) if self.last_step is not None else None
        for name, target in (("best", best), ("final", final)):
            if target is None:
                continue
            link = self.root / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target.name)
        return {"best_step": self.best_step, "best_combined_val_loss": self.best_loss,
                "best_checkpoint": str(best) if best else None, "final_step": self.last_step,
                "final_checkpoint": str(final) if final else None, "kept_checkpoints": sorted(str(p) for p in self.saved.values())}


# --------------------------------------------------------------------------- main

def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t_start = time.time()
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = pick_device(cfg.get("device", "auto"))
    precision = cfg["precision"]
    if precision == "bf16" and device == "cpu":
        log.warning("bf16 autocast on CPU is slow; consider precision=fp32 for CPU smoke tests")
    if cfg.get("loss_type", "sigmoid") != "sigmoid":
        raise ValueError("Arm 1 is vanilla sigmoid DPO only")
    if not cfg.get("full_finetune", True):
        raise ValueError("Arm 1 requires full-parameter fine-tuning (no LoRA)")

    run_dir = Path(cfg["runs_dir"]) / f"{cfg['run_name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    batch = resolve_batch_config(cfg)
    lengths = resolve_lengths(cfg)
    resolved = {**{k: v for k, v in cfg.items() if not k.startswith("_")}, "resolved_batch": batch,
                "resolved_lengths": lengths, "device": device, "torch_version": torch.__version__}
    write_json(run_dir / "config_resolved.json", resolved)
    log.info("run_dir=%s device=%s batch=%s lengths=%s", run_dir, device, batch, lengths)

    # ---- data
    tokenizer = load_tokenizer(cfg["model"])
    train_rows = load_split_rows(cfg["artifacts_dir"], "train")
    val_rows = load_split_rows(cfg["artifacts_dir"], "val")
    for r in train_rows[:200] + val_rows[:50]:
        assert_no_objective_leakage(tokenizer, r)
    n_h = sum(1 for r in train_rows if r["objective"] == "helpfulness")
    n_s = len(train_rows) - n_h
    log.info("train rows=%d (helpfulness=%d harmlessness=%d) val rows=%d", len(train_rows), n_h, n_s, len(val_rows))
    if n_h != n_s:
        log.warning("train rows are not exactly balanced between objectives: H=%d S=%d", n_h, n_s)
    t0 = time.time()
    train_enc = encode_rows(tokenizer, train_rows, lengths["max_prompt_length"], lengths["max_length"], on_overflow="error")
    val_enc = encode_rows(tokenizer, val_rows, lengths["max_prompt_length"], lengths["max_length"], on_overflow="error")
    log.info("tokenized in %.1fs", time.time() - t0)
    pad_id = tokenizer.pad_token_id

    # ---- model (fp32 master weights)
    model = load_model(cfg["model"], device, torch.float32, gradient_checkpointing=bool(cfg["gradient_checkpointing"]))
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("model %s: %.1fM params, %.1fM trainable (full fine-tuning)", cfg["model"], n_params / 1e6, n_trainable / 1e6)
    assert n_trainable == n_params, "full-parameter fine-tuning expected"

    # ---- reference log-probs (model at init == frozen reference)
    hashes = read_json(Path(cfg["artifacts_dir"]) / "splits" / "manifest_hashes.json")
    eval_bs = int(cfg["eval_batch_size"])
    ref: Dict[str, Dict[str, Dict[str, float]]] = {}
    ref_info = {}
    for split, enc in (("train", train_enc), ("val", val_enc)):
        key = ref_cache_key(cfg["model"], hashes[split]["dpo_rows_sha256"], lengths["max_prompt_length"], lengths["max_length"], device, precision)
        cache_dir = cfg.get("ref_cache_dir") if cfg.get("precompute_ref_logprobs", True) else None
        t0 = time.time()
        ref[split], cached = get_or_compute_ref_logprobs(
            cache_dir, split, key, lambda enc=enc, split=split: compute_logprobs(model, enc, pad_id, eval_bs, device, precision, f"ref:{split}"))
        ref_info[split] = {"cache_key": key, "from_cache": cached, "seconds": time.time() - t0}
    write_json(run_dir / "ref_logprobs_info.json", ref_info)

    # ---- optimizer / schedule
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), betas=(float(cfg["adam_beta1"]), float(cfg["adam_beta2"])),
        eps=float(cfg["adam_epsilon"]), weight_decay=float(cfg["weight_decay"]))
    n_rows = len(train_enc)
    steps_per_epoch = math.ceil(n_rows / batch["global_batch_size"])
    total_steps = steps_per_epoch * int(cfg["max_epochs"])
    if cfg.get("max_steps"):
        total_steps = min(total_steps, int(cfg["max_steps"]))
    scheduler, warmup_steps = build_scheduler(optimizer, cfg, total_steps)
    log.info("steps/epoch=%d total_steps=%d warmup=%d", steps_per_epoch, total_steps, warmup_steps)

    beta, ls = float(cfg["beta"]), float(cfg.get("label_smoothing", 0.0))
    ckpt = CheckpointManager(run_dir / "checkpoints", cfg["checkpoint_keep"], cfg["checkpoint_dtype"])
    metrics_f = open(run_dir / "metrics.jsonl", "w", encoding="utf-8")
    val_history: List[Dict[str, Any]] = []

    def evaluate(step: int, epoch: float, is_epoch_end: bool) -> Dict[str, Any]:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)  # free gradient memory before the no-grad validation pass
        free_cache(device)
        pol = compute_logprobs(model, val_enc, pad_id, eval_bs, device, precision, "val")
        recs = row_records(val_rows, pol, ref["val"], beta, ls)
        m = summarize(recs, beta)
        m.update({"step": step, "epoch": epoch, "is_epoch_end": is_epoch_end, "seconds": time.time() - t0})
        is_best = ckpt.save(model, tokenizer, step, epoch, m, is_epoch_end)
        m["is_best"] = is_best
        val_history.append(m)
        write_json(run_dir / "val_history.json", val_history)
        metrics_f.write(json.dumps({"type": "val", **{k: v for k, v in m.items() if k != "groups"},
                                    "groups": {g: {kk: vv for kk, vv in s.items()} for g, s in m["groups"].items()}}) + "\n")
        metrics_f.flush()
        g = m["groups"]
        log.info("VAL step=%d epoch=%.2f combined=%.4f H=%.4f S=%.4f agree=%.4f conflict=%.4f acc=%.3f margin=%.4f best=%s",
                 step, epoch, m["combined_loss"], g["helpfulness"]["loss"], g["harmlessness"]["loss"],
                 g["agreement"]["loss"], g["conflict"]["loss"], g["overall"]["pref_acc"], g["overall"]["mean_margin"], is_best)
        return m

    # ---- step-0 sanity: policy == reference -> margins 0, loss log(2)
    m0 = evaluate(0, 0.0, False)
    if abs(m0["groups"]["overall"]["mean_margin"]) > 1e-3 or abs(m0["combined_loss"] - math.log(2)) > 1e-3:
        log.warning("step-0 validation is not exactly log(2)/zero-margin (loss=%.5f margin=%.5f); "
                    "expected when reference log-probs came from a cache computed on another device/precision",
                    m0["combined_loss"], m0["groups"]["overall"]["mean_margin"])

    # ---- training loop
    model.train()
    step, eval_every, log_every = 0, int(cfg["eval_every_steps"]), int(cfg.get("log_every", 10))
    done = False
    for epoch in range(int(cfg["max_epochs"])):
        order = epoch_order(n_rows, seed, epoch)
        for gstart in range(0, n_rows, batch["global_batch_size"]):
            gidx = order[gstart:gstart + batch["global_batch_size"]]
            n_in_global = len(gidx)
            optimizer.zero_grad(set_to_none=True)
            acc = {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0, "loss_H": 0.0, "n_H": 0, "loss_S": 0.0, "n_S": 0,
                   "chosen_reward": 0.0, "rejected_reward": 0.0}
            for mstart in range(0, n_in_global, batch["per_device_batch_size"]):
                midx = gidx[mstart:mstart + batch["per_device_batch_size"]]
                rows = [train_rows[i] for i in midx]
                b = collate_dpo([train_enc[i] for i in midx], pad_id)
                out = forward_logprobs(model, b, device, precision)
                ref_c = torch.tensor([ref["train"][r["row_id"]]["chosen"] for r in rows], device=device)
                ref_r = torch.tensor([ref["train"][r["row_id"]]["rejected"] for r in rows], device=device)
                d = dpo_loss(out["chosen_logps"], out["rejected_logps"], ref_c, ref_r, beta, ls)
                # Equal weighting of every preference row; the sum over the global batch is divided by
                # the number of rows actually in it (the last batch of an epoch may be smaller).
                (d["loss"].sum() / n_in_global).backward()
                with torch.no_grad():
                    acc["loss"] += float(d["loss"].sum()); acc["margin"] += float(d["margin"].sum())
                    acc["acc"] += float((d["margin"] > 0).sum()); acc["n"] += len(rows)
                    acc["chosen_reward"] += float(d["chosen_reward"].sum()); acc["rejected_reward"] += float(d["rejected_reward"].sum())
                    for j, r in enumerate(rows):
                        k = "H" if r["objective"] == "helpfulness" else "S"
                        acc[f"loss_{k}"] += float(d["loss"][j]); acc[f"n_{k}"] += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_f = epoch + (gstart + n_in_global) / n_rows
            rec = {"type": "train", "step": step, "epoch": epoch_f, "lr": scheduler.get_last_lr()[0],
                   "loss": acc["loss"] / acc["n"], "margin": acc["margin"] / acc["n"], "pref_acc": acc["acc"] / acc["n"],
                   "chosen_reward": acc["chosen_reward"] / acc["n"], "rejected_reward": acc["rejected_reward"] / acc["n"],
                   "loss_helpfulness": (acc["loss_H"] / acc["n_H"]) if acc["n_H"] else None,
                   "loss_harmlessness": (acc["loss_S"] / acc["n_S"]) if acc["n_S"] else None,
                   "n_rows_H": acc["n_H"], "n_rows_S": acc["n_S"], "grad_norm": float(grad_norm), "elapsed_s": time.time() - t_start}
            metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
            if step % log_every == 0 or step == 1:
                log.info("step %d/%d ep %.2f loss %.4f (H %.4f S %.4f) acc %.3f margin %.4f gn %.3f lr %.2e",
                         step, total_steps, epoch_f, rec["loss"], rec["loss_helpfulness"] or 0, rec["loss_harmlessness"] or 0,
                         rec["pref_acc"], rec["margin"], rec["grad_norm"], rec["lr"])
            is_epoch_end = gstart + n_in_global >= n_rows
            if step >= total_steps:
                done = True
            if step % eval_every == 0 or is_epoch_end or done:
                evaluate(step, epoch_f, is_epoch_end or done)
                model.train()
            if done:
                break
        if done:
            break

    metrics_f.close()
    summary = {
        "run_dir": str(run_dir), "seed": seed, "model": cfg["model"], "total_steps": step, "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps, "batch": batch, "lengths": lengths, "beta": beta, "label_smoothing": ls,
        "train_rows": n_rows, "train_rows_helpfulness": n_h, "train_rows_harmlessness": n_s, "val_rows": len(val_rows),
        "elapsed_s": time.time() - t_start, "ref_logprobs": ref_info,
        "val_curve": [{"step": m["step"], "epoch": m["epoch"], "combined_loss": m["combined_loss"],
                       "helpfulness_loss": m["groups"]["helpfulness"]["loss"], "harmlessness_loss": m["groups"]["harmlessness"]["loss"],
                       "agreement_loss": m["groups"]["agreement"]["loss"], "conflict_loss": m["groups"]["conflict"]["loss"],
                       "pref_acc": m["groups"]["overall"]["pref_acc"], "is_best": m["is_best"]} for m in val_history],
        **ckpt.finalize(),
    }
    write_json(run_dir / "train_summary.json", summary)
    best = next(m for m in val_history if m["step"] == summary["best_step"])
    (run_dir / "train_summary.md").write_text(
        f"# Training summary ({run_dir.name})\n\nbest step {summary['best_step']} (combined val loss {summary['best_combined_val_loss']:.4f})\n\n"
        + render_metrics_markdown(best, f"validation at best step {summary['best_step']}"), encoding="utf-8")
    log.info("done: best step %s combined val loss %.4f (%.1fs)", summary["best_step"], summary["best_combined_val_loss"], summary["elapsed_s"])
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--seed", type=int, default=None, help="training seed (overrides config)")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    train(cfg)


if __name__ == "__main__":
    main()
