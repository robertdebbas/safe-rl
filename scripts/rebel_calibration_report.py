"""Summarize artifacts/rebel/calibration_<arm>.json: mu/sigma (+SE) per head, d_std/d_raw, reference thresholds, attainability."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
art = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "artifacts") / "rebel"
rows = ["| arm | head | mu (SE) | sigma (SE) | p5 / median / p95 | d_std | d_raw | alpha=0 / 1 raw | frac pi0 resp <= d | frac prompts best-of-4 <= d |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for f in sorted(art.glob("calibration_*.json")):
    c = json.loads(f.read_text())
    ref = c.get("reference_thresholds", {})
    att = c.get("attainability_under_pi0", {})
    for h, s in c["heads"].items():
        is_cost = h == "cost"
        d_std = c.get("d_std") if is_cost else ""
        d_raw = "%.4f" % c["d_raw"] if is_cost and c.get("d_raw") is not None else ""
        refs = ("%.3f / %.3f" % (ref["alpha=0.0"]["d_raw"], ref["alpha=1.0"]["d_raw"])) if is_cost and ref else ""
        a1 = att.get("frac_responses_at_or_below_d", "") if is_cost else ""
        a2 = att.get("frac_prompts_best_of_4_at_or_below_d", "") if is_cost else ""
        rows.append("| %s | %s | %.4f (%.4f) | %.4f (%.4f) | %.2f / %.2f / %.2f | %s | %s | %s | %s | %s |" % (
            c["arm"], h, s["mu"], s["mu_se"], s["sigma"], s["sigma_se"], s["p5"], s["median"], s["p95"], d_std, d_raw, refs, a1, a2))
    print("%s: n=%d generations_sha=%s reward-cost spearman(pi0)=%s" % (c["arm"], c["n_trajectories"], c["generations_sha256"][:12], c.get("reward_cost_spearman_pi0")))
md = "\n".join(rows)
print(md)
(art / "calibration_report.md").write_text(md + "\n")
