"""PRESPECIFIED hyperparameter selection from the pilots (training signals + the held-out development set; never a test prompt).

Runs: runs/sixarm_hpval_<family>_<arm>_lr<lr>_eta<eta>_g<gamma>_seed42  (family rebel | grpo; GRPO pilots carry eta 1.0 / g 0 as placeholders).
Per run, from rollout_batches.jsonl (40 batches under the actual schedule):
  finite    every record finite                       fit_frac  (REBEL) fraction of batches whose post-step regression loss < pre-step loss
  kl_prev   (REBEL) mean per-token KL(pi_t || pi_{t+1})          clip_frac (GRPO) mean clipped-token fraction (per-step movement proxy)
and from hpval_dev_eval.json (Beaver on the 256 held-out development prompts at start/end):
  dev_help / dev_harm deltas, trunc_final, kl0_final, modeled_reward delta.
Gates (prespecified): G1 fit_frac >= 0.80 (REBEL only)   G2 kl_prev <= 1e-3 (REBEL) / clip_frac <= 0.05 (GRPO)   G3 trunc_final <= 0.25
G4 kl0_final <= 1.0   G5 no exploitation: NOT (modeled reward rose AND dev helpfulness fell by > 0.5)   G6 dev harmfulness rose by <= 0.5
REBEL rule: largest lr passing every gate at eta 1, gamma 0 (none -> smallest lr, flagged); keep eta 1 unless eta in {0.5, 2} passes every
gate and raises fit_frac by >= 0.10; adopt gamma 0.02 (token-mean) only if its run passes every gate and lowers kl0_final; the constrained
pilot at the selected settings must pass G1-G4 and G6 (it cannot change the selection).
GRPO rule: largest lr in {3e-7, 1e-6, 3e-6} passing every gate (none -> 1e-6, the previously validated value, flagged).
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

GATES = {"fit_frac_min": 0.80, "kl_prev_max": 1e-3, "clip_frac_max": 0.05, "trunc_max": 0.25, "kl0_max": 1.0, "help_drop_max": 0.5, "harm_rise_max": 0.5}
NAME_RE = re.compile(r"sixarm_hpval_(?P<family>rebel|grpo)_(?P<arm>[a-z0-9_]+?)_lr(?P<lr>[0-9.e+-]+)_eta(?P<eta>[0-9.]+)_g(?P<g>[0-9.]+)_seed")


def summarize(run: Path) -> dict:
    m = NAME_RE.search(run.name)
    rows = [json.loads(l) for l in open(run / "rollout_batches.jsonl")]
    rows = sorted({r["batch_index"]: r for r in rows}.values(), key=lambda r: r["batch_index"])
    s = {"run": run.name, "family": m["family"], "arm": m["arm"], "lr": float(m["lr"]), "eta": float(m["eta"]), "gamma": float(m["g"]), "n_batches": len(rows),
         "finite": all(r.get("finite", True) for r in rows), "grad_norm": statistics.fmean(g for r in rows for g in r["grad_norm"]),
         "trunc_last5": statistics.fmean(r["truncation_rate"] for r in rows[-5:]), "len_last5": statistics.fmean(r["completion_len_mean"] for r in rows[-5:])}
    if m["family"] == "rebel":
        fit = [r["post_loss"] < r["loss_at_theta_t"] for r in rows if "post_loss" in r]
        s.update({"fit_frac": statistics.fmean(fit) if fit else float("nan"), "kl_prev": statistics.fmean(r["kl_prev"]["kl_k3_token_mean"] for r in rows if "kl_prev" in r),
                  "post_corr": statistics.fmean(r.get("post_pred_target_corr", float("nan")) for r in rows),
                  "kl_shaping_pair_abs_mean": statistics.fmean(r.get("kl_shaping_pair_abs_mean", 0.0) for r in rows), "pair_signal_abs_mean": statistics.fmean(r.get("pair_signal_abs_mean", 0.0) for r in rows)})
    else:
        s.update({"clip_frac": statistics.fmean(r["clip_fraction"] for r in rows), "ratio_mean": statistics.fmean(r["ratio_mean"] for r in rows),
                  "kl_ref_last5": statistics.fmean(r["kl_mean"] for r in rows[-5:])})
    dev = run / "hpval_dev_eval.json"
    if dev.exists():
        d = json.loads(dev.read_text())
        s.update({"dev_help_delta": d["delta"]["beaver_helpfulness"], "dev_harm_delta": d["delta"]["beaver_harmfulness"], "trunc_final": d["final"]["truncation_rate"],
                  "kl0_final": d["final"]["kl_initial_token"], "modeled_reward_delta": d["delta"]["modeled_reward"], "dev_scored": True})
    else:
        s["dev_scored"] = False
    return s


def gates(s: dict) -> dict:
    g = {"finite": bool(s["finite"])}
    if s["family"] == "rebel":
        g["G1_fit"] = s["fit_frac"] >= GATES["fit_frac_min"]; g["G2_kl_prev"] = s["kl_prev"] <= GATES["kl_prev_max"]
    else:
        g["G2_clip"] = s["clip_frac"] <= GATES["clip_frac_max"]
    if s.get("dev_scored"):
        g.update({"G3_trunc": s["trunc_final"] <= GATES["trunc_max"], "G4_kl0": s["kl0_final"] <= GATES["kl0_max"],
                  "G5_no_exploitation": not (s["modeled_reward_delta"] > 0 and s["dev_help_delta"] < -GATES["help_drop_max"]),
                  "G6_no_harm_drift": s["dev_harm_delta"] <= GATES["harm_rise_max"]})
    else:
        g["dev_checks"] = "MISSING (scripts/sixarm_hpval_devscore.py)"
    return g


def passed(s: dict, skip=()) -> bool:
    return all(v is True for k, v in gates(s).items() if k not in skip)


def select(summaries: list[dict]) -> dict:
    out = {"gates": GATES}
    sc = [s for s in summaries if s["family"] == "rebel" and s["arm"] == "scalar"]
    base = [s for s in sc if s["eta"] == 1.0 and s["gamma"] == 0.0]
    if base:
        ok = [s for s in base if passed(s)]
        best = max(ok, key=lambda s: s["lr"]) if ok else min(base, key=lambda s: s["lr"])
        reason = "largest lr passing every gate at eta=1, gamma=0" if ok else "NO eta=1 REBEL pilot passed every gate; smallest lr chosen -- REVIEW before launch"
        lr, eta, gamma = best["lr"], 1.0, 0.0
        for s in sc:
            if s["lr"] == lr and s["gamma"] == 0.0 and s["eta"] != 1.0 and passed(s) and s["fit_frac"] >= best["fit_frac"] + 0.10:
                eta, best, reason = s["eta"], s, f"eta {s['eta']} passes every gate and improves fit_frac by >= 0.10"
        for s in sc:
            if s["lr"] == lr and s["eta"] == eta and s["gamma"] > 0 and passed(s) and s.get("kl0_final", 9e9) < best.get("kl0_final", 9e9):
                gamma, reason = s["gamma"], reason + f"; gamma {s['gamma']} (token-mean) adopted: passes every gate and lowers KL to pi_0"
        cons = [s for s in summaries if s["family"] == "rebel" and s["arm"] != "scalar" and s["lr"] == lr and s["eta"] == eta and s["gamma"] == gamma]
        out["rebel"] = {"learning_rate": lr, "rebel_eta": eta, "kl_to_initial_coef": gamma, "reason": reason, "chosen": best,
                        "constrained_check": {"runs": [s["run"] for s in cons], "passed": all(passed(s, skip=("G5_no_exploitation",)) for s in cons)} if cons else {"runs": [], "passed": None, "note": "not run yet"}}
    gr = [s for s in summaries if s["family"] == "grpo"]
    if gr:
        ok = [s for s in gr if passed(s)]
        best = max(ok, key=lambda s: s["lr"]) if ok else next((s for s in gr if s["lr"] == 1e-6), min(gr, key=lambda s: s["lr"]))
        out["grpo"] = {"learning_rate": best["lr"], "reason": "largest lr passing every gate" if ok else "NO GRPO pilot passed every gate; 1e-6 (previously validated) kept -- REVIEW", "chosen": best}
    return out


if __name__ == "__main__":
    runs = sorted(Path("runs").glob("sixarm_hpval_*_seed42"))
    summaries = [summarize(r) for r in runs if (r / "rollout_batches.jsonl").exists() and NAME_RE.search(r.name)]
    for s in summaries:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in s.items()}, gates(s))
    if summaries:
        sel = select(summaries)
        Path("runs/sixarm_hpval_selection.json").write_text(json.dumps({"summaries": summaries, "selection": sel}, indent=2))
        print("SELECTED:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "chosen"} for k, v in sel.items() if k != "gates"}, indent=1))
    else:
        print("no pilot runs found", file=sys.stderr)
