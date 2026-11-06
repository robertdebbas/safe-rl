"""Write the pilot-selected hyperparameters into the six arm configs (runs/sixarm_hpval_selection.json -> configs/sixarm/arm*.yaml).
Appends an explicit, dated override block; the arm files inherit everything else unchanged. Refuses to write if the selection is
missing or flagged for review (exit 3) so an unattended chain stops instead of training with unvalidated values.
    python scripts/sixarm_apply_selection.py [--stage 1|2]   (stage 1: GRPO lr + provisional REBEL lr; stage 2: final REBEL lr/eta/gamma)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REBEL = ["arm0_rebel_separate_d1", "arm1_rebel_separate_d0", "arm2_rebel_scalar"]
GRPO = ["arm3_grpo_separate_d1", "arm4_grpo_separate_d0", "arm5_grpo_scalar"]
MARK = "# --- selected by the pilots"


def write_block(name: str, values: dict, stage: int) -> None:
    p = ROOT / "configs" / "sixarm" / f"{name}.yaml"
    txt = p.read_text()
    if MARK in txt:
        txt = txt[: txt.index(MARK)].rstrip() + "\n"
    block = f"{MARK} (stage {stage}, {dt.datetime.now().isoformat(timespec='minutes')}; rule: scripts/sixarm_hpval_select.py) ---\n"
    block += "".join(f"{k}: {v}\n" for k, v in values.items())
    p.write_text(txt + block)
    print(f"{p.name}: {values}")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, default=2); a = ap.parse_args()
    sel_path = ROOT / "runs" / "sixarm_hpval_selection.json"
    if not sel_path.exists():
        print("no selection file", file=sys.stderr); raise SystemExit(3)
    sel = json.loads(sel_path.read_text())["selection"]
    if "rebel" not in sel or "grpo" not in sel:
        print(f"selection incomplete: {list(sel)}", file=sys.stderr); raise SystemExit(3)
    for fam in ("rebel", "grpo"):
        if "REVIEW" in sel[fam]["reason"]:
            print(f"{fam} selection flagged for review: {sel[fam]['reason']}", file=sys.stderr); raise SystemExit(3)
    if a.stage == 2 and sel["rebel"]["constrained_check"].get("passed") is not True:
        print(f"constrained check did not pass: {sel['rebel']['constrained_check']}", file=sys.stderr); raise SystemExit(3)
    for n in GRPO:
        write_block(n, {"learning_rate": sel["grpo"]["learning_rate"]}, a.stage)
    for n in REBEL:
        write_block(n, {"learning_rate": sel["rebel"]["learning_rate"], "rebel_eta": sel["rebel"]["rebel_eta"], "kl_to_initial_coef": sel["rebel"]["kl_to_initial_coef"]}, a.stage)


if __name__ == "__main__":
    main()
