#!/usr/bin/env bash
# 1) generate the common 256 x 4 pi_0 calibration responses once (shared by all arms); 2) score them with the arm's frozen scorer(s)
# and write artifacts/rebel/calibration_<arm>.json (mu, sigma per head; d_std / d_raw and reference thresholds for the constrained arms).
#   bash scripts/run_rebel_calibrate.sh scalar | shared | separate
set -euo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?arm: scalar | shared | separate}"
export PYTHONPATH=src
case "$ARM" in scalar) CONFIG=configs/rebel_scalar.yaml;; shared) CONFIG=configs/rebel_shared_constrained.yaml;; separate) CONFIG=configs/rebel_separate_constrained.yaml;; *) echo "unknown arm"; exit 1;; esac
test -f artifacts/rebel/train_prompts.jsonl || bash scripts/run_rebel_prepare.sh
python3 -m saferl.rebel_calibrate --config "$CONFIG" --stage both
