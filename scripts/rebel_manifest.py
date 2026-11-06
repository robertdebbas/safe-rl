"""Reproducible run manifest for the REBEL study: hashes of code, configs, prompt manifests, calibration files and frozen scorer
checkpoints, plus library versions. Writes runs/rebel_run_manifest.json. Read-only."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
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

    code = sorted(list((ROOT / "src" / "saferl").glob("rebel_*.py")) + [ROOT / "src" / "saferl" / "train_rebel.py", ROOT / "src" / "saferl" / "grpo_matched.py",
                  ROOT / "src" / "saferl" / "grpo_rollout.py", ROOT / "src" / "saferl" / "grpo_scoring.py", ROOT / "src" / "saferl" / "dpo_loss.py",
                  ROOT / "src" / "saferl" / "eval_generate.py", ROOT / "src" / "saferl" / "eval_score.py", ROOT / "src" / "saferl" / "eval_merge.py"])
    cfgs = sorted(list((ROOT / "configs").glob("rebel_*.yaml")) + list((ROOT / "configs").glob("grpo_matched_*.yaml")) + [ROOT / "configs" / "eval.yaml", ROOT / "configs" / "eval_rebel.yaml"])
    art = ROOT / "artifacts" / "rebel"
    data = sorted(art.glob("*.jsonl")) + sorted(art.glob("*.json")) if art.exists() else []
    scorers = {}
    for cal in sorted(art.glob("calibration_*.json")) if art.exists() else []:
        scorers[cal.name] = json.loads(cal.read_text()).get("scorer_checkpoints")
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip() or None
    except Exception:
        git = None
    m = {"git_head": git, "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__,
         "code": {str(p.relative_to(ROOT)): sha(p) for p in code}, "configs": {str(p.relative_to(ROOT)): sha(p) for p in cfgs},
         "artifacts_rebel": {str(p.relative_to(ROOT)): sha(p) for p in data}, "scorer_checkpoints_from_calibration": scorers,
         "eval_manifest": sha(ROOT / "artifacts" / "eval" / "eval_prompts.jsonl") if (ROOT / "artifacts" / "eval" / "eval_prompts.jsonl").exists() else None}
    out = ROOT / "runs" / "rebel_run_manifest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else f"{len(v)} entries") for k, v in m.items()}, indent=1))


if __name__ == "__main__":
    main()
