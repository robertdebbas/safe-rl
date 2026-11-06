"""Reproducibility manifest for the six-arm study: hashes of code, the six configs, prompt artifacts (exposure list, hpval set),
calibration thresholds, frozen scorer fingerprints, library versions. Writes runs/sixarm_run_manifest.json. Read-only."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import torch
    import transformers

    code = [ROOT / "src" / "saferl" / n for n in ("train_rebel.py", "rebel_core.py", "rebel_data.py", "train_grpo.py", "grpo_core.py", "grpo_rollout.py",
                                                    "grpo_scoring.py", "grpo_data.py", "dpo_loss.py", "eval_common.py", "eval_generate.py", "eval_score.py", "eval_merge.py")]
    cfgs = sorted((ROOT / "configs" / "sixarm").glob("*.yaml")) + [ROOT / "configs" / n for n in ("rebel_common.yaml", "grpo_common.yaml", "eval.yaml", "eval_sixarm.yaml")]
    arts = [p for p in (ROOT / "artifacts" / "sixarm").rglob("*") if p.is_file()] + [p for p in (ROOT / "artifacts" / "sixarm_hpval").rglob("*") if p.is_file()]
    thresholds = {}
    for cal in sorted((ROOT / "artifacts" / "sixarm").glob("calibration_separate_d*.json")):
        c = json.loads(cal.read_text())
        thresholds[cal.name] = {"d": c["d"], "rule": c.get("d_rule"), "d_se": c.get("uncertainty", {}).get("d_se"), "scorer_checkpoints": c.get("scorer_checkpoints")}
    sel = ROOT / "runs" / "sixarm_hpval_selection.json"
    m = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__,
         "code": {str(p.relative_to(ROOT)): sha(p) for p in code if p.exists()}, "configs": {str(p.relative_to(ROOT)): sha(p) for p in cfgs if p.exists()},
         "artifacts": {str(p.relative_to(ROOT)): sha(p) for p in arts}, "thresholds": thresholds,
         "hyperparameter_selection": json.loads(sel.read_text()).get("selection") if sel.exists() else None,
         "eval_manifest": sha(ROOT / "artifacts" / "eval" / "eval_prompts.jsonl") if (ROOT / "artifacts" / "eval" / "eval_prompts.jsonl").exists() else None}
    out = ROOT / "runs" / "sixarm_run_manifest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else f"{len(v)} entries") for k, v in m.items()}, indent=1))


if __name__ == "__main__":
    main()
