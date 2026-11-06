#!/usr/bin/env python3
"""Prepare audited, prompt-disjoint, fixed splits from the official PKU-SafeRLHF dataset.

    python scripts/prepare_data.py --config configs/data.yaml
    python scripts/prepare_data.py --config configs/data_smoke.yaml
    python scripts/prepare_data.py --config configs/data.yaml --set out_dir=/tmp/regen   # regeneration check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from saferl.prepare import run_prepare  # noqa: E402
from saferl.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. out_dir=/tmp/x")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    report = run_prepare(cfg)
    print("\n=== SUMMARY ===")
    for s, st in report["official_agreement"].items():
        print(f"official {s}: n={st['n']} agreement={st['agreement']} ({st['agreement_pct']:.4f}%) conflict={st['conflict']} ({st['conflict_pct']:.4f}%)")
    print("length limits:", report["length_limits"])
    for n, s in report["splits"].items():
        ac = s["summary"]["agreement_conflict"]
        print(f"{n}: pairs={ac['n']} agreement={ac['agreement']} conflict={ac['conflict']} dpo_rows={s['n_dpo_rows']}")
    print("manifest hashes:", {k: v["manifest_sha256"][:16] for k, v in report["manifest_hashes"].items()})


if __name__ == "__main__":
    main()
