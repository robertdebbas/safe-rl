#!/usr/bin/env bash
# Six-arm study submission (Kempner AI cluster). NOTHING is submitted unless this script is invoked with a stage, and every stage
# needs explicit approval. GPU jobs: 6 pilots + 4 pilots + 6 training + 6 evaluation = 22. CPU-only work never takes a GPU.
#   bash scripts/slurm/submit_sixarm.sh check      # login node, read-only: account, partitions, GPU types, limits, idle GPUs, disk
#   bash scripts/slurm/submit_sixarm.sh calibrate  # login node CPU: thresholds d0/d1 + prompt artifacts + manifest (seconds)
#   bash scripts/slurm/submit_sixarm.sh pilot1     # 6 GPU jobs: REBEL lr {1e-6,3e-6,1e-5} (eta 1, gamma 0) + GRPO lr {3e-7,1e-6,3e-6};
#                                                  #   each job = 40-batch training + in-job held-out dev scoring (also the engineering smoke)
#   bash scripts/slurm/submit_sixarm.sh select     # login node CPU: apply the prespecified rule -> runs/sixarm_hpval_selection.json
#   HPVAL_LR=<lr> bash scripts/slurm/submit_sixarm.sh pilot2   # 4 GPU jobs: eta 0.5 / 2.0, gamma 0.02, constrained check at the selected lr
#   bash scripts/slurm/submit_sixarm.sh train      # 6 GPU jobs, concurrent (after the selected values are written into the arm configs)
#   bash scripts/slurm/submit_sixarm.sh eval       # 6 GPU jobs (arm k on all 1,960 prompts; arm 0's job also evaluates base) -> 1 CPU job
#                                                  #   (merge + 2,000-replicate standard and paired bootstraps + manifest)
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?stage}"
ACCOUNT="${ACCOUNT:-kempner_kdbrantley_lab}"
PARTITION="${PARTITION:-kempner_h100}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-16}"
MEM="${MEM:-96G}"
CONSTRAINT="${CONSTRAINT:-}"
CPU_PARTITION="${CPU_PARTITION:-shared}"          # FASRC CPU partition for CPU-only work (never a Kempner GPU partition)
CPU_ACCOUNT="${CPU_ACCOUNT:-calmon_lab}"
T_PILOT="${T_PILOT:-02:30:00}"; T_TRAIN="${T_TRAIN:-08:00:00}"; T_EVAL="${T_EVAL:-05:00:00}"
mkdir -p runs; JOBS=runs/slurm_jobs_sixarm.tsv; touch "$JOBS"
ARMS=(arm0_rebel_separate_d1 arm1_rebel_separate_d0 arm2_rebel_scalar arm3_grpo_separate_d1 arm4_grpo_separate_d0 arm5_grpo_scalar)

sb() {  # GPU job: name time dependency export script
  local name="$1" time="$2" dep="$3" exp="$4" script="$5"
  local args=(--parsable --job-name="$name" --time="$time" --partition="$PARTITION" --account="$ACCOUNT" --gres="$GRES"
              --cpus-per-task="$CPUS" --mem="$MEM" --export="ALL,$exp")
  [ -n "$CONSTRAINT" ] && args+=(--constraint="$CONSTRAINT")
  [ -n "$dep" ] && args+=(--dependency="$dep")
  local id; id=$(sbatch "${args[@]}" "$script")
  printf '%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$id" "$name" "$dep" >> "$JOBS"; echo "$id"
}
sb_cpu() {  # CPU job: name time dependency command
  local name="$1" time="$2" dep="$3" cmd="$4"
  local args=(--parsable --job-name="$name" --time="$time" --partition="$CPU_PARTITION" --account="$CPU_ACCOUNT" --export="ALL,CMD=$cmd")
  [ -n "$dep" ] && args+=(--dependency="$dep")
  local id; id=$(sbatch "${args[@]}" scripts/slurm/cpu.sbatch)
  printf '%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$id" "$name" "$dep" >> "$JOBS"; echo "$id"
}

case "$STAGE" in
  check)
    bash scripts/slurm/kempner_check.sh ;;
  calibrate)
    PYTHONPATH=src python3 scripts/sixarm_calibration.py && PYTHONPATH=src python3 scripts/sixarm_prepare.py && PYTHONPATH=src python3 scripts/sixarm_manifest.py ;;
  pilot1)
    for LR in 1e-6 3e-6 1e-5; do sb "pilot_rebel_lr${LR}" "$T_PILOT" "" "SCRIPT=scripts/run_sixarm_hpval.sh,ARGS=rebel $LR 1.0 0 scalar" scripts/slurm/generic_1gpu.sbatch; done
    for LR in 3e-7 1e-6 3e-6; do sb "pilot_grpo_lr${LR}" "$T_PILOT" "" "SCRIPT=scripts/run_sixarm_hpval.sh,ARGS=grpo $LR" scripts/slurm/generic_1gpu.sbatch; done ;;
  select)
    PYTHONPATH=src python3 scripts/sixarm_hpval_select.py ;;
  pilot2)
    LR0="${HPVAL_LR:?set HPVAL_LR to the REBEL lr selected in pilot1}"; ETA0="${HPVAL_ETA:-1.0}"
    for ETA in 0.5 2.0; do sb "pilot_rebel_lr${LR0}_eta${ETA}" "$T_PILOT" "" "SCRIPT=scripts/run_sixarm_hpval.sh,ARGS=rebel $LR0 $ETA 0 scalar" scripts/slurm/generic_1gpu.sbatch; done
    sb "pilot_rebel_lr${LR0}_g0.02" "$T_PILOT" "" "SCRIPT=scripts/run_sixarm_hpval.sh,ARGS=rebel $LR0 $ETA0 0.02 scalar" scripts/slurm/generic_1gpu.sbatch
    for G in 0 0.02; do sb "pilot_rebel_sep_d1_g$G" "$T_PILOT" "" "SCRIPT=scripts/run_sixarm_hpval.sh,ARGS=rebel $LR0 $ETA0 $G separate_d1" scripts/slurm/generic_1gpu.sbatch; done ;;
  auto)         # unattended chain: a CPU job that runs scripts/slurm/sixarm_auto.sh <step> after the dependency CPU_DEP
    STEP="${2:?after_pilot1 | after_pilot2 | after_train}"
    sb_cpu "sixarm_auto_$STEP" "00:30:00" "${CPU_DEP:-}" "bash scripts/slurm/sixarm_auto.sh $STEP" ;;
  train)
    for A in "${ARMS[@]}"; do sb "sixarm_$A" "$T_TRAIN" "" "ARM_CONFIG=configs/sixarm/$A.yaml,SEED=42" scripts/slurm/sixarm_train.sbatch; done ;;
  eval)
    IDS=""
    for K in 0 1 2 3 4 5; do
      WB=""; [ "$K" = 0 ] && WB="--with-base"
      id=$(sb "eval_arm$K" "$T_EVAL" "" "SCRIPT=scripts/run_sixarm_eval.sh,ARGS=$K $WB" scripts/slurm/generic_1gpu.sbatch); IDS="${IDS:+$IDS:}$id"
    done
    sb_cpu eval_merge_pairs_manifest "00:40:00" "afterok:$IDS" \
      "PYTHONPATH=src python3 -m saferl.eval_merge --config configs/eval_sixarm.yaml --seeds 42 --methods base sixarm_arm0 sixarm_arm1 sixarm_arm2 sixarm_arm3 sixarm_arm4 sixarm_arm5 && PYTHONPATH=src python3 scripts/eval_pairs.py --config configs/eval_sixarm.yaml && PYTHONPATH=src python3 scripts/sixarm_manifest.py" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "done ($STAGE); see $JOBS"
