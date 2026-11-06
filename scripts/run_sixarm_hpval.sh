#!/usr/bin/env bash
# One pilot = training (40 batches, actual schedule) followed IN THE SAME JOB by the held-out development scoring (Beaver on the
# 256 monitoring samples at start and end). The pilot is also the engineering smoke: it exercises the full code path at real scale.
#   bash scripts/run_sixarm_hpval.sh rebel <lr> <eta> <gamma> [scalar|separate_d1]
#   bash scripts/run_sixarm_hpval.sh grpo  <lr>
set -euo pipefail
cd "$(dirname "$0")/.."
FAMILY="${1:?rebel | grpo}"
export PYTHONPATH=src WORLD_SIZE=1
case "$FAMILY" in
  rebel)
    LR="${2:?lr}"; ETA="${3:?eta}"; GAMMA="${4:-0}"; ARM="${5:-scalar}"
    case "$ARM" in scalar) CFG=configs/sixarm/hpval_rebel_scalar.yaml;; separate_d1) CFG=configs/sixarm/hpval_rebel_separate_d1.yaml;; *) echo "arm: scalar | separate_d1"; exit 1;; esac
    NAME="sixarm_hpval_rebel_${ARM}_lr${LR}_eta${ETA}_g${GAMMA}"
    python3 -m saferl.train_rebel --config "$CFG" --seed 42 --set learning_rate="$LR" rebel_eta="$ETA" kl_to_initial_coef="$GAMMA" run_name="$NAME" ;;
  grpo)
    LR="${2:?lr}"
    NAME="sixarm_hpval_grpo_scalar_lr${LR}_eta1.0_g0"
    python3 -m saferl.train_grpo --config configs/sixarm/hpval_grpo_scalar.yaml --seed 42 --set learning_rate="$LR" run_name="$NAME" ;;
  *) echo "family: rebel | grpo"; exit 1 ;;
esac
python3 scripts/sixarm_hpval_devscore.py "runs/${NAME}_seed42"
