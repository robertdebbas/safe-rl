"""Reward / cost model trainer for Architectures A (scalar), B (shared two-head) and C
(separate_reward / separate_cost). Reuses the DPO infrastructure: manifests, Qwen formatting and
tokenization, length limits, batch/accumulation resolution, bf16 autocast with fp32 master weights,
checkpoint manager, seeds, and metric conventions.

Per optimizer step: a global batch of ``global_batch_size`` comparison rows (each = chosen and
rejected sequence), split into micro-batches; one backbone forward per micro-batch over 2B
sequences; each row's loss uses only the head of its objective, so unused heads get no gradient.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .formatting import assert_no_objective_leakage, collate_dpo
from .modeling import compute_logprobs  # noqa: F401  (kept import surface identical to DPO)
from .modeling import encode_rows, load_tokenizer
from .rm_data import ARCHITECTURES, load_rm_rows
from .rm_loss import bt_loss, cost_loss
from .rm_metrics import render_metrics_markdown, row_records, summarize
from .rm_models import ScoreModel, decay_param_groups
from .train_dpo import CheckpointManager, build_scheduler, epoch_order, free_cache, resolve_batch_config, resolve_lengths
from .utils import get_logger, load_config, pick_device, seed_everything, write_json

log = get_logger("saferl.train_rm")

COMMON_HPARAM_KEYS = (
    "cost_pointwise_classification",
    "model", "full_finetune", "max_epochs", "global_batch_size", "learning_rate", "optimizer", "adam_beta1", "adam_beta2",
    "adam_epsilon", "weight_decay", "weight_decay_exclude_bias_and_norm", "lr_scheduler", "warmup_ratio", "max_grad_norm",
    "precision", "gradient_checkpointing", "eval_every_steps", "checkpoint_keep", "checkpoint_dtype", "max_prompt_length",
    "max_length", "data_seed",
)


def arch_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    arch = cfg["arch"]
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHITECTURES)}")
    spec = dict(ARCHITECTURES[arch])
    # Pointwise safe/unsafe classification of the cost head is an OPTIONAL ablation; OFF in primary configs.
    spec["cost_pointwise_classification"] = bool(cfg.get("cost_pointwise_classification", False))
    return spec


def save_score_model(model: ScoreModel, tokenizer, path: Path, dtype: str, state: Dict[str, Any]) -> None:
    model.save_pretrained(path, dtype)
    tokenizer.save_pretrained(path)
    write_json(path / "trainer_state.json", state)


def compute_scores(model: ScoreModel, encoded_rows, rows, pad_id: int, batch_size: int, device: str, precision: str,
                   head_for: Dict[str, str], desc: str = "scores") -> Dict[str, Dict[str, float]]:
    """Chosen/rejected scores per row under the head of the row's objective (no grad, eval mode)."""
    from .modeling import autocast_ctx

    was_training = model.training
    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    order = sorted(range(len(encoded_rows)), key=lambda i: max(len(encoded_rows[i]["chosen"]["input_ids"]),
                                                               len(encoded_rows[i]["rejected"]["input_ids"])))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            b = collate_dpo([encoded_rows[i] for i in idx], pad_id)
            B = b["batch_size"]
            with autocast_ctx(device, precision):
                h = model.hidden_last(b["input_ids"].to(device), b["attention_mask"].to(device))
            heads_needed = sorted({head_for[rows[i]["objective"]] for i in idx})
            sc = model.apply_heads(h, heads_needed)
            for j, i in enumerate(idx):
                hd = head_for[rows[i]["objective"]]
                results[rows[i]["row_id"]] = {"chosen": float(sc[hd][j]), "rejected": float(sc[hd][B + j]), "head": hd}
            if (start // batch_size) % 50 == 0:
                log.info("%s: %d/%d rows", desc, min(start + batch_size, len(order)), len(order))
    if was_training:
        model.train()
    return results


def microbatch_loss(model: ScoreModel, batch: Dict[str, Any], rows: List[Dict[str, Any]], spec: Dict[str, Any],
                    device: str, precision: str) -> Dict[str, Any]:
    """Per-row losses for one micro-batch; only the head of each row's objective is applied."""
    from .modeling import autocast_ctx

    B = batch["batch_size"]
    with autocast_ctx(device, precision):
        h = model.hidden_last(batch["input_ids"].to(device), batch["attention_mask"].to(device))
    per_row_loss = torch.zeros(B, device=device)
    per_row_margin = torch.zeros(B, device=device)
    parts: Dict[str, Dict[str, float]] = {}
    for objective in spec["objectives"]:
        idx = [i for i, r in enumerate(rows) if r["objective"] == objective]
        if not idx:
            continue
        head = spec["head_for"][objective]
        it = torch.tensor(idx, device=device)
        sc = model.apply_heads(h, [head])[head]
        s_chosen, s_rejected = sc[it], sc[B + it]
        if spec["loss_for"][objective] == "bt":
            out = bt_loss(s_chosen, s_rejected)
            parts[objective] = {"loss": float(out["loss"].sum()), "n": len(idx)}
        else:
            if spec.get("cost_pointwise_classification", False):  # ablation only
                sign_c = torch.tensor([rows[i]["chosen_sign"] for i in idx], device=device, dtype=torch.float32)
                sign_r = torch.tensor([rows[i]["rejected_sign"] for i in idx], device=device, dtype=torch.float32)
                out = cost_loss(s_chosen, s_rejected, sign_c, sign_r, pointwise_classification=True)
                parts[objective] = {"loss": float(out["loss"].sum()), "rank_loss": float(out["rank_loss"].sum()),
                                    "class_loss": float(out["class_loss"].sum()), "n": len(idx)}
            else:  # PRIMARY: pairwise BT on relative safety only (chosen = safer_response_id); labels unused
                out = cost_loss(s_chosen, s_rejected)
                parts[objective] = {"loss": float(out["loss"].sum()), "n": len(idx)}
        per_row_loss[it] = out["loss"].float()
        per_row_margin[it] = out["margin"].float()
    return {"loss": per_row_loss, "margin": per_row_margin, "parts": parts}


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t_start = time.time()
    spec = arch_spec(cfg)
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = pick_device(cfg.get("device", "auto"))
    precision = cfg["precision"]
    if not cfg.get("full_finetune", True):
        raise ValueError("reward models require full-parameter fine-tuning (no LoRA)")
    run_dir = Path(cfg["runs_dir"]) / f"{cfg['run_name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    batch = resolve_batch_config(cfg)
    lengths = resolve_lengths(cfg)
    resolved = {**{k: v for k, v in cfg.items() if not k.startswith("_")}, "resolved_batch": batch, "resolved_lengths": lengths,
                "device": device, "torch_version": torch.__version__, "arch_spec": spec}
    write_json(run_dir / "config_resolved.json", resolved)
    log.info("arch=%s run_dir=%s device=%s batch=%s lengths=%s", cfg["arch"], run_dir, device, batch, lengths)

    # ---- data: the SAME manifests as DPO, objectives per architecture
    tokenizer = load_tokenizer(cfg["model"])
    train_rows = load_rm_rows(cfg["artifacts_dir"], "train", spec["objectives"])
    val_rows = load_rm_rows(cfg["artifacts_dir"], "val", spec["objectives"])
    for r in train_rows[:200] + val_rows[:50]:
        assert_no_objective_leakage(tokenizer, r)
    counts = {o: sum(1 for r in train_rows if r["objective"] == o) for o in spec["objectives"]}
    log.info("train rows=%d %s val rows=%d", len(train_rows), counts, len(val_rows))
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"objectives are not equally represented: {counts}")
    train_enc = encode_rows(tokenizer, train_rows, lengths["max_prompt_length"], lengths["max_length"], "error")
    val_enc = encode_rows(tokenizer, val_rows, lengths["max_prompt_length"], lengths["max_length"], "error")
    pad_id = tokenizer.pad_token_id
    train_tokens = sum(len(e["chosen"]["input_ids"]) + len(e["rejected"]["input_ids"]) for e in train_enc)

    # ---- model
    model = ScoreModel.from_backbone(cfg["model"], spec["heads"], device, torch.float32, bool(cfg["gradient_checkpointing"]))
    n_params = model.num_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == n_params["total"], "full fine-tuning expected"
    log.info("model %s heads=%s params=%s", cfg["model"], spec["heads"], n_params)

    # ---- optimizer / schedule
    wd = float(cfg["weight_decay"])
    groups = decay_param_groups(model, wd) if cfg.get("weight_decay_exclude_bias_and_norm", True) else \
        [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": wd}]
    optimizer = torch.optim.AdamW(groups, lr=float(cfg["learning_rate"]), betas=(float(cfg["adam_beta1"]), float(cfg["adam_beta2"])),
                                  eps=float(cfg["adam_epsilon"]))
    n_rows = len(train_enc)
    steps_per_epoch = math.ceil(n_rows / batch["global_batch_size"])
    total_steps = steps_per_epoch * int(cfg["max_epochs"])
    if cfg.get("max_steps"):
        total_steps = min(total_steps, int(cfg["max_steps"]))
    scheduler, warmup_steps = build_scheduler(optimizer, cfg, total_steps)
    log.info("steps/epoch=%d total_steps=%d warmup=%d", steps_per_epoch, total_steps, warmup_steps)

    ckpt = CheckpointManager(run_dir / "checkpoints", cfg["checkpoint_keep"], cfg["checkpoint_dtype"], save_fn=save_score_model)
    metrics_f = open(run_dir / "metrics.jsonl", "w", encoding="utf-8")
    val_history: List[Dict[str, Any]] = []
    eval_bs = int(cfg["eval_batch_size"])

    def evaluate(step: int, epoch: float, is_epoch_end: bool) -> Dict[str, Any]:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        free_cache(device)
        sc = compute_scores(model, val_enc, val_rows, pad_id, eval_bs, device, precision, spec["head_for"], "val")
        m = summarize(row_records(val_rows, sc, spec["loss_for"], spec["cost_pointwise_classification"]))
        m.update({"step": step, "epoch": epoch, "is_epoch_end": is_epoch_end, "seconds": time.time() - t0})
        is_best = ckpt.save(model, tokenizer, step, epoch, m, is_epoch_end)
        m["is_best"] = is_best
        val_history.append(m)
        write_json(run_dir / "val_history.json", val_history)
        metrics_f.write(json.dumps({"type": "val", **m}) + "\n"); metrics_f.flush()
        g = m["groups"]
        log.info("VAL step=%d epoch=%.2f combined=%.4f H=%s S=%s acc=%.3f best=%s", step, epoch, m["combined_loss"],
                 f"{g['helpfulness']['loss']:.4f}" if g["helpfulness"]["n"] else "-",
                 f"{g['harmlessness']['loss']:.4f}" if g["harmlessness"]["n"] else "-", g["overall"]["acc_tie_adjusted"], is_best)
        return m

    evaluate(0, 0.0, False)
    model.train()
    step, eval_every, log_every = 0, int(cfg["eval_every_steps"]), int(cfg.get("log_every", 10))
    done = False
    for epoch in range(int(cfg["max_epochs"])):
        order = epoch_order(n_rows, seed, epoch)  # independent shuffle of all rows (H and S not co-located)
        for gstart in range(0, n_rows, batch["global_batch_size"]):
            gidx = order[gstart:gstart + batch["global_batch_size"]]
            n_in_global = len(gidx)
            optimizer.zero_grad(set_to_none=True)
            acc: Dict[str, float] = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
            parts_acc: Dict[str, Dict[str, float]] = {}
            for mstart in range(0, n_in_global, batch["per_device_batch_size"]):
                midx = gidx[mstart:mstart + batch["per_device_batch_size"]]
                rows = [train_rows[i] for i in midx]
                b = collate_dpo([train_enc[i] for i in midx], pad_id)
                out = microbatch_loss(model, b, rows, spec, device, precision)
                (out["loss"].sum() / n_in_global).backward()
                with torch.no_grad():
                    acc["loss"] += float(out["loss"].sum()); acc["margin"] += float(out["margin"].sum())
                    acc["acc"] += float((out["margin"] > 0).sum()); acc["n"] += len(rows)
                    for o, p in out["parts"].items():
                        pa = parts_acc.setdefault(o, {})
                        for k, v in p.items():
                            pa[k] = pa.get(k, 0.0) + v
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            optimizer.step(); scheduler.step(); step += 1
            epoch_f = epoch + (gstart + n_in_global) / n_rows
            rec = {"type": "train", "step": step, "epoch": epoch_f, "lr": scheduler.get_last_lr()[0],
                   "loss": acc["loss"] / acc["n"], "acc": acc["acc"] / acc["n"], "margin": acc["margin"] / acc["n"],
                   "grad_norm": float(grad_norm), "elapsed_s": time.time() - t_start}
            for o, p in parts_acc.items():
                rec[f"n_rows_{o}"] = int(p["n"])
                for k, v in p.items():
                    if k != "n":
                        rec[f"{k}_{o}"] = v / p["n"]
            metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
            if step % log_every == 0 or step == 1:
                log.info("step %d/%d ep %.2f loss %.4f acc %.3f margin %.3f gn %.3f lr %.2e %s", step, total_steps, epoch_f,
                         rec["loss"], rec["acc"], rec["margin"], rec["grad_norm"], rec["lr"],
                         {o: round(p["loss"] / p["n"], 4) for o, p in parts_acc.items()})
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

    elapsed = time.time() - t_start
    summary = {
        "run_dir": str(run_dir), "arch": cfg["arch"], "seed": seed, "model": cfg["model"], "heads": spec["heads"],
        "objectives": spec["objectives"], "total_steps": step, "steps_per_epoch": steps_per_epoch, "warmup_steps": warmup_steps,
        "batch": batch, "lengths": lengths, "train_rows": n_rows, "train_rows_per_objective": counts, "val_rows": len(val_rows),
        "parameters": n_params, "train_tokens_per_epoch": train_tokens,
        "approx_train_flops": 6.0 * n_params["total"] * train_tokens * (step / steps_per_epoch),
        "elapsed_s": elapsed,
        "val_curve": [{"step": m["step"], "epoch": m["epoch"], "combined_loss": m["combined_loss"],
                       "helpfulness_loss": m["groups"]["helpfulness"].get("loss"), "harmlessness_loss": m["groups"]["harmlessness"].get("loss"),
                       "acc_tie_adjusted": m["groups"]["overall"]["acc_tie_adjusted"], "is_best": m["is_best"]} for m in val_history],
        **ckpt.finalize(),
    }
    write_json(run_dir / "train_summary.json", summary)
    best = next(m for m in val_history if m["step"] == summary["best_step"])
    (run_dir / "train_summary.md").write_text(
        f"# Training summary ({run_dir.name}, arch={cfg['arch']})\n\nbest step {summary['best_step']} "
        f"(combined val loss {summary['best_combined_val_loss']:.4f}); params {n_params['total']:,}; "
        f"tokens/epoch {train_tokens:,}; approx FLOPs {summary['approx_train_flops']:.3e}; wallclock {elapsed:.0f}s\n\n"
        + render_metrics_markdown(best, f"validation at best step {summary['best_step']}"), encoding="utf-8")
    log.info("done: best step %s combined val loss %.4f (%.0fs)", summary["best_step"], summary["best_combined_val_loss"], elapsed)
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    train(cfg)


if __name__ == "__main__":
    main()
