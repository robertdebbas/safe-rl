#!/usr/bin/env python3
"""Initialization-only diagnostic for the score models (no training).

For each architecture (scalar F; shared R/C; separate R; separate C), reconstructed exactly as the
trainer does (seed_everything(42) then ScoreModel.from_backbone), report on a fixed deterministic
subset of its own training rows:
  * per head: score mean / std / p5 / p95 / max |score| over all scored sequences;
  * initial BT loss per objective (pairwise, as in the primary configs);
  * initial pre-clip gradient norm of the actual first global batch of the seed-42 epoch order.

    python scripts/rm_init_diagnostic.py --config configs/rm_common.yaml            # 1.5B, as in science configs
    python scripts/rm_init_diagnostic.py --config configs/rm_common.yaml --grad_dtype bf16   # low-memory machines
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import saferl  # noqa: E402,F401
from saferl.formatting import collate_dpo  # noqa: E402
from saferl.modeling import autocast_ctx, encode_rows, load_tokenizer  # noqa: E402
from saferl.rm_data import ARCHITECTURES, load_rm_rows  # noqa: E402
from saferl.rm_models import ScoreModel  # noqa: E402
from saferl.train_dpo import epoch_order, resolve_lengths  # noqa: E402
from saferl.train_rm import arch_spec, microbatch_loss  # noqa: E402
from saferl.utils import get_logger, load_config, pick_device, seed_everything, write_json  # noqa: E402

log = get_logger("saferl.rm_init_diag")


def stats(x: np.ndarray) -> dict:
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()), "p5": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)), "max_abs": float(np.abs(x).max()), "min": float(x.min()), "max": float(x.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rm_common.yaml")
    ap.add_argument("--archs", nargs="*", default=["scalar", "shared", "separate_reward", "separate_cost"])
    ap.add_argument("--n_score_rows", type=int, default=256, help="rows (x2 sequences) for score statistics")
    ap.add_argument("--grad_dtype", choices=["fp32", "bf16"], default="fp32",
                    help="parameter dtype for the gradient-norm pass (fp32 = exactly as training; bf16 = low memory)")
    ap.add_argument("--grad_offload", choices=["none", "cpu"], default="cpu",
                    help="cpu: stream each parameter's gradient to an fp32 CPU accumulator as soon as it is produced, so the "
                         "device never holds a full gradient copy (exact same gradient; needed on small-memory devices)")
    ap.add_argument("--grad_device", default=None,
                    help="device for the gradient-norm pass (default: same as scoring). 'cpu' runs that pass in fp32 on the "
                         "CPU for machines whose accelerator cannot hold weights + gradients of the 1.5B model")
    ap.add_argument("--out", default="artifacts/rm_init_diagnostic")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    device = pick_device(cfg.get("device", "auto"))
    precision = cfg["precision"]
    lengths = resolve_lengths(cfg)
    tok = load_tokenizer(cfg["model"])
    gbs, pdbs = int(cfg["global_batch_size"]), int(cfg["per_device_batch_size"])
    report = {"model": cfg["model"], "device": device, "precision": precision, "grad_dtype": args.grad_dtype,
              "global_batch_size": gbs, "per_device_batch_size": pdbs, "n_score_rows": args.n_score_rows,
              "cost_pointwise_classification": bool(cfg.get("cost_pointwise_classification", False)), "archs": {}}

    for arch in args.archs:
        t0 = time.time()
        spec = arch_spec({**cfg, "arch": arch})
        rows = load_rm_rows(cfg["artifacts_dir"], "train", spec["objectives"])
        order = epoch_order(len(rows), 42, 0)
        sub_idx = order[: args.n_score_rows]
        batch_idx = order[:gbs]  # the real first global batch of seed 42
        enc = {i: e for i, e in zip(sub_idx + batch_idx, encode_rows(tok, [rows[i] for i in sub_idx + batch_idx],
                                                                       lengths["max_prompt_length"], lengths["max_length"], "error"))}
        seed_everything(42)  # identical head initialization to the real run
        if device == "mps":
            torch.mps.empty_cache()
        try:
            model = ScoreModel.from_backbone(cfg["model"], spec["heads"], device, torch.float32, bool(cfg["gradient_checkpointing"]))
        except Exception as e:  # noqa: BLE001
            log.error("%s: model construction failed: %r", arch, e)
            report["archs"][arch] = {"error": repr(e)}
            continue
        model.eval()
        params = model.num_parameters()
        head_norms = {n: {"weight_norm": float(h.weight.norm()), "bias": float(h.bias.item())} for n, h in model.heads.items()}

        # ---- score statistics + initial BT loss on the subset (eval mode, bf16 autocast as in validation)
        scores = {n: [] for n in spec["heads"]}
        per_obj_loss = {o: [] for o in spec["objectives"]}
        per_obj_margin = {o: [] for o in spec["objectives"]}
        with torch.no_grad():
            for start in range(0, len(sub_idx), 8):
                idx = sub_idx[start:start + 8]
                b = collate_dpo([enc[i] for i in idx], tok.pad_token_id)
                B = b["batch_size"]
                with autocast_ctx(device, precision):
                    h = model.hidden_last(b["input_ids"].to(device), b["attention_mask"].to(device))
                sc = model.apply_heads(h)
                for n in spec["heads"]:
                    scores[n].extend(sc[n].float().cpu().tolist())
                for j, i in enumerate(idx):
                    o = rows[i]["objective"]
                    hd = spec["head_for"][o]
                    m = float(sc[hd][j] - sc[hd][B + j])  # chosen - rejected == harmful - safer for cost rows? no:
                    if spec["loss_for"][o] == "cost":
                        m = float(sc[hd][B + j] - sc[hd][j])  # harmful - safer
                    per_obj_margin[o].append(m)
                    per_obj_loss[o].append(float(np.logaddexp(0.0, -m)))
        arch_rep = {
            "heads": spec["heads"], "objectives": spec["objectives"], "parameters": params, "head_init": head_norms,
            "n_train_rows": len(rows), "scores": {n: stats(np.array(v)) for n, v in scores.items()},
            "initial_bt_loss": {o: float(np.mean(v)) for o, v in per_obj_loss.items()},
            "initial_margin": {o: stats(np.array(v)) for o, v in per_obj_margin.items()},
            "initial_rank_acc": {o: float(np.mean(np.array(v) > 0)) for o, v in per_obj_margin.items()},
        }
        log.info("%s scores=%s bt_loss=%s", arch, {n: (round(s['mean'], 3), round(s['std'], 3)) for n, s in arch_rep["scores"].items()},
                 arch_rep["initial_bt_loss"])

        # ---- gradient norm of the first global batch (pre-clip), exactly the trainer's micro-batching
        report["archs"][arch] = arch_rep
        write_json(Path(args.out).with_suffix(".json"), report)  # incremental: score stats survive a crash below
        try:
            gdev = args.grad_device or device
            gprec = precision if args.grad_dtype == "fp32" else "bf16"
            if gdev != device:
                model.cpu()  # pure device move (no dtype change)
                if device == "mps":
                    torch.mps.empty_cache()
                if gdev == "cpu":
                    gprec = "fp32"  # CPU bf16 autocast is impractically slow; gradient norm then reflects fp32 numerics
                model.to(gdev)
            if args.grad_dtype == "bf16":
                model.cpu(); model.to(torch.bfloat16); model.to(gdev)  # separate ops (see modeling.save_checkpoint note)
            model.train()
            model.zero_grad(set_to_none=True)
            acc, handles = {}, []
            if args.grad_offload == "cpu":
                acc = {n: torch.zeros(p.shape, dtype=torch.float32, device="cpu") for n, p in model.named_parameters()}

                def make_hook(name):
                    def hook(p):
                        acc[name].add_(p.grad.detach().float().cpu())  # same dtype move (no fused device+dtype cast)
                        p.grad = None
                    return hook

                handles = [p.register_post_accumulate_grad_hook(make_hook(n)) for n, p in model.named_parameters()]
            n_in_global = len(batch_idx)
            total_loss = 0.0
            comp = {o: 0 for o in spec["objectives"]}
            for mstart in range(0, n_in_global, pdbs):
                midx = batch_idx[mstart:mstart + pdbs]
                mrows = [rows[i] for i in midx]
                for r in mrows:
                    comp[r["objective"]] += 1
                b = collate_dpo([enc[i] for i in midx], tok.pad_token_id)
                out = microbatch_loss(model, b, mrows, spec, gdev, gprec)
                (out["loss"].sum() / n_in_global).backward()
                total_loss += float(out["loss"].sum())
                del out, b
                if gdev == "mps":
                    torch.mps.empty_cache()
                log.info("%s grad pass: %d/%d rows", arch, min(mstart + pdbs, n_in_global), n_in_global)
            for h in handles:
                h.remove()
            sq = 0.0
            head_sq = {n: 0.0 for n in spec["heads"]}
            grads = acc.items() if acc else ((n, p.grad) for n, p in model.named_parameters() if p.grad is not None)
            for name, g_t in grads:
                g = float(g_t.detach().float().norm()) ** 2
                sq += g
                for n in spec["heads"]:
                    if name.startswith(f"heads.{n}."):
                        head_sq[n] += g
            del acc
            arch_rep["first_batch"] = {"n_rows": n_in_global, "composition": comp, "mean_loss": total_loss / n_in_global,
                                       "grad_norm_total": float(np.sqrt(sq)), "grad_norm_backbone": float(np.sqrt(sq - sum(head_sq.values()))),
                                       "grad_norm_heads": {n: float(np.sqrt(v)) for n, v in head_sq.items()}, "grad_dtype": args.grad_dtype,
                                       "grad_device": gdev, "grad_precision": gprec, "clip_at": float(cfg["max_grad_norm"])}
            log.info("%s first-batch loss=%.4f grad_norm=%.2f (backbone %.2f, heads %s) [%s] %.0fs", arch, arch_rep["first_batch"]["mean_loss"],
                     arch_rep["first_batch"]["grad_norm_total"], arch_rep["first_batch"]["grad_norm_backbone"],
                     {n: round(v, 3) for n, v in arch_rep["first_batch"]["grad_norm_heads"].items()}, args.grad_dtype, time.time() - t0)
        except Exception as e:  # noqa: BLE001
            log.error("%s gradient pass failed: %r", arch, e)
            arch_rep["first_batch"] = {"error": repr(e), "grad_dtype": args.grad_dtype}
        arch_rep["seconds"] = time.time() - t0
        report["archs"][arch] = arch_rep
        del model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    write_json(out.with_suffix(".json"), report)
    md = [f"# Score-model initialization diagnostic ({cfg['model']}, device {device}, grad pass in {args.grad_dtype})\n",
          f"score subset: first {args.n_score_rows} rows of each architecture's seed-42 epoch order (x2 sequences); "
          f"gradient batch: the real first global batch of {gbs} rows, micro-batch {pdbs}, pre-clip norm (clip at {cfg['max_grad_norm']}).\n",
          "| arch | head | n seqs | mean | std | p5 | p95 | max abs |", "|---|---|---|---|---|---|---|---|"]
    for a, r in report["archs"].items():
        if "scores" not in r:
            md.append(f"| {a} | failed: {r.get('error', '')[:60]} | | | | | | |")
            continue
        for n, s in r["scores"].items():
            md.append(f"| {a} | {n} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | {s['p5']:.4f} | {s['p95']:.4f} | {s['max_abs']:.4f} |")
    md += ["", "| arch | objective | initial BT loss | mean margin | rank acc | first-batch loss | grad norm total | backbone | heads |",
           "|---|---|---|---|---|---|---|---|---|"]
    for a, r in report["archs"].items():
        if "first_batch" not in r:
            continue
        fb = r["first_batch"]
        ok = "error" not in fb
        for o in r["objectives"]:
            md.append(f"| {a} | {o} | {r['initial_bt_loss'][o]:.4f} | {r['initial_margin'][o]['mean']:.4f} | {r['initial_rank_acc'][o]:.3f} | "
                      + (f"{fb['mean_loss']:.4f} | {fb['grad_norm_total']:.3f} | {fb['grad_norm_backbone']:.3f} | "
                         f"{ {n: round(v, 4) for n, v in fb['grad_norm_heads'].items()} } |" if ok else f"failed: {fb['error'][:80]} | - | - | - |"))
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
