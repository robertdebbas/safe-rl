#!/usr/bin/env bash
# REBEL pipeline on Slurm (FASRC Cannon). NOTHING here runs unless this script is invoked explicitly (launch requires approval).
#   bash scripts/slurm/submit_rebel.sh prepare      # CPU, login node: build artifacts/rebel manifests (no GPU job)
#   bash scripts/slurm/submit_rebel.sh calibrate    # pi_0 calibration generations -> per-arm mu/sigma/d (GPU, ~30 min total)
#   bash scripts/slurm/submit_rebel.sh smoke        # 1.5B 3-batch smoke per arm (GPU, ~15 min each)
#   bash scripts/slurm/submit_rebel.sh chain42      # per arm: 100-batch gate (exit 2 on hard failure) -> continuation -> 4 eval shards -> merge
#   bash scripts/slurm/submit_rebel.sh matched42    # PREPARED matched constrained-GRPO comparison (shared, separate), same chain
#   bash scripts/slurm/submit_rebel.sh manifest     # write runs/rebel_run_manifest.json (hashes of code, configs, data, calibration)
# Env knobs: ACCOUNT (helpers/eval), ACCOUNT_TRAIN (training), WIDE_PARTITION, CONSTRAINT, EXCLUDE, T_* time limits.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?stage}"
ACCOUNT="${ACCOUNT:-calmon_lab}"
ACCOUNT_TRAIN="${ACCOUNT_TRAIN:-kdbrantley_lab}"
WIDE_PARTITION="${WIDE_PARTITION:-gpu,gpu_requeue,seas_gpu,gpu_h200}"
CONSTRAINT="${CONSTRAINT:-a100|h100|h200}"
EXCLUDE="${EXCLUDE:-holygpu8a31105,holygpu8a12204}"
T_CAL="${T_CAL:-02:00:00}"; T_SMOKE="${T_SMOKE:-01:30:00}"; T_DIAG="${T_DIAG:-04:00:00}"; T_TRAIN="${T_TRAIN:-20:00:00}"; T_EVAL="${T_EVAL:-06:00:00}"
GPUS=1; MEM="${MEM:-96G}"
mkdir -p runs; JOBS=runs/slurm_jobs_rebel.tsv; touch "$JOBS"

sb() {  # name time partition dependency export script [account]
  local name="$1" time="$2" part="$3" dep="$4" exp="$5" script="$6" acct="${7:-$ACCOUNT}"
  local args=(--parsable --job-name="$name" --time="$time" --partition="$part" --account="$acct" --gres="gpu:$GPUS" --mem="$MEM"
              --constraint="$CONSTRAINT" --exclude="$EXCLUDE" --export="ALL,$exp")
  [ -n "$dep" ] && args+=(--dependency="$dep")
  local id; id=$(sbatch "${args[@]}" "$script")
  printf '%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$id" "$name" "$dep" >> "$JOBS"; echo "$id"
}

case "$STAGE" in
  prepare)
    bash scripts/run_rebel_prepare.sh ;;
  calibrate)
    GEN=$(sb rebel_cal_gen "$T_CAL" "$WIDE_PARTITION" "" "PY=1,SCRIPT=-m saferl.rebel_calibrate,ARGS=--config configs/rebel_scalar.yaml --stage generate" scripts/slurm/generic.sbatch)
    for A in scalar shared separate; do
      case "$A" in scalar) C=configs/rebel_scalar.yaml;; shared) C=configs/rebel_shared_constrained.yaml;; separate) C=configs/rebel_separate_constrained.yaml;; esac
      sb "rebel_cal_$A" "00:40:00" "$WIDE_PARTITION" "afterok:$GEN" "PY=1,SCRIPT=-m saferl.rebel_calibrate,ARGS=--config $C --stage score" scripts/slurm/generic.sbatch
    done ;;
  smoke)
    for A in scalar shared separate; do sb "rebel_smoke_$A" "$T_SMOKE" "$WIDE_PARTITION" "" "SCRIPT=scripts/run_rebel_smoke_1p5b.sh,ARGS=$A" scripts/slurm/generic.sbatch; done ;;
  chain42|matched42)
    MATCHED=0; ARMS="scalar shared separate"; PREFIX=rebel; METHOD_PREFIX=rebel
    if [ "$STAGE" = matched42 ]; then MATCHED=1; ARMS="shared separate"; PREFIX=grpo_matched; METHOD_PREFIX=grpo_matched; fi
    EVAL_IDS=""
    for ARM in $ARMS; do
      DIAG=$(sb "${PREFIX}_${ARM}_s42_diag100" "$T_DIAG" "$WIDE_PARTITION" "" "MATCHED=$MATCHED,ARM=$ARM,SEED=42,EXTRA=--set stop_after_batches=100" scripts/slurm/rebel_seed.sbatch)
      CONT=$(sb "${PREFIX}_${ARM}_s42" "$T_TRAIN" "$WIDE_PARTITION" "afterok:$DIAG" "MATCHED=$MATCHED,ARM=$ARM,SEED=42" scripts/slurm/rebel_seed.sbatch "$ACCOUNT_TRAIN")
      for I in 0 1 2 3; do
        id=$(sb "eval_${PREFIX}_${ARM}_s42_$I" "$T_EVAL" "$WIDE_PARTITION" "afterok:$CONT" "METHOD=${METHOD_PREFIX}_$ARM,SEED=42,N=4,I=$I,CONFIG=configs/eval_rebel.yaml" scripts/slurm/eval_shard.sbatch)
        EVAL_IDS="${EVAL_IDS:+$EVAL_IDS:}$id"
      done
    done
    sb "eval_merge_${PREFIX}_s42" "00:20:00" "$WIDE_PARTITION" "afterok:$EVAL_IDS" "PY=1,SCRIPT=-m saferl.eval_merge,ARGS=--config configs/eval_rebel.yaml --seeds 42" scripts/slurm/generic.sbatch ;;
  manifest)
    PYTHONPATH=src python3 scripts/rebel_manifest.py ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "submitted; see $JOBS"
