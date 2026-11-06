#!/usr/bin/env bash
# Unattended orchestration of the six-arm study: each step runs as a CPU job after the previous GPU stage and submits the next one.
#   after_pilot1 : selection rule -> provisional values -> submit pilot2 (5 GPU jobs) -> submit after_pilot2 (CPU, afterany)
#   after_pilot2 : selection rule -> final values written into the arm configs + manifest -> submit train (6 GPU) -> after_train (CPU, afterok)
#   after_train  : submit eval (6 GPU jobs, then the CPU merge / bootstraps / manifest)
# Any failed selection (flagged for review, constrained check failed) stops the chain with exit 3; nothing trains on unvalidated values.
set -euo pipefail
cd "$(dirname "$0")/../.."
STEP="${1:?after_pilot1 | after_pilot2 | after_train}"
export PYTHONPATH=src
SUB=scripts/slurm/submit_sixarm.sh
JOBS=runs/slurm_jobs_sixarm.tsv

case "$STEP" in
  after_pilot1)
    python3 scripts/sixarm_hpval_select.py
    python3 scripts/sixarm_apply_selection.py --stage 1
    LR=$(python3 -c "import json; print(json.load(open('runs/sixarm_hpval_selection.json'))['selection']['rebel']['learning_rate'])")
    IDS=$(HPVAL_LR="$LR" bash "$SUB" pilot2 | grep -E '^[0-9]+$' | paste -sd: -)
    echo "pilot2 jobs: $IDS (REBEL lr $LR)"
    CPU_DEP="afterok:$IDS" bash "$SUB" auto after_pilot2 ;;
  after_pilot2)
    python3 scripts/sixarm_hpval_select.py
    python3 scripts/sixarm_apply_selection.py --stage 2
    python3 scripts/sixarm_manifest.py
    IDS=$(bash "$SUB" train | grep -E '^[0-9]+$' | paste -sd: -)
    echo "train jobs: $IDS"
    CPU_DEP="afterok:$IDS" bash "$SUB" auto after_train ;;
  after_train)
    bash "$SUB" eval ;;
  *) echo "unknown step $STEP"; exit 1 ;;
esac
