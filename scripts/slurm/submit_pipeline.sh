#!/usr/bin/env bash
# Submit the frozen experiment as a Slurm dependency DAG. Run ON THE CLUSTER from the repo root.
#
#   bash scripts/slurm/submit_pipeline.sh smoke          # 1.5B smokes: DPO, RM x4 (seed 42 data), eval pilot  (no full runs)
#   bash scripts/slurm/submit_pipeline.sh rm42           # the four seed-42 RM jobs (critical path)
#   bash scripts/slurm/submit_pipeline.sh grpo-chain     # calibration -> report -> GRPO smokes -> full GRPO seeds, dependent on rm42 job ids
#   bash scripts/slurm/submit_pipeline.sh rest           # RM seeds 43/44 and DPO seeds 42/43/44 (parallel, off the critical path)
#   bash scripts/slurm/submit_pipeline.sh eval           # sharded evaluation of all methods (after their final checkpoints exist)
#
# Resource knobs (env): ACCOUNT, PARTITION (non-preemptible, default gpu), WIDE_PARTITION (default gpu,gpu_requeue), CONSTRAINT, EXCLUDE,
# GPUS (1 or 2 -> only model placement, trainers stay single-process), times T_RM/T_DPO/T_GRPO/T_EVAL/T_SMOKE (realistic, backfill friendly).
# Job ids are appended to runs/slurm_jobs.tsv so dependencies and status can be tracked.
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?stage}"
# Partition policy (FASRC Cannon): 80 GB+ GPUs only. `gpu` = A100-80GB, non-preemptible (RM/DPO full runs, which do not
# resume). `gpu,gpu_requeue` = widest 80GB+ pool incl. H100/H200, preemptible with --requeue (smokes, calibration, resumable
# GRPO, idempotent eval shards). Sub-80GB / MIG / A40 / A6000 nodes are excluded explicitly; `gpu_test` is MIG-only (unusable).
ACCOUNT="${ACCOUNT:-kdbrantley_lab}"                       # helpers / evaluation
ACCOUNT_TRAIN="${ACCOUNT_TRAIN:-$ACCOUNT}"                # long training jobs (RM, DPO, GRPO) may use a different account
PARTITION="${PARTITION:-gpu,seas_gpu,gpu_h200}"                          # non-preemptible 80GB+ pools (RM, DPO): A100-80 / A100-80 (SEAS) / H200
WIDE_PARTITION="${WIDE_PARTITION:-gpu,gpu_requeue,seas_gpu,gpu_h200}"    # + preemptible gpu_requeue for restartable work
CONSTRAINT="${CONSTRAINT:-a100|h100|h200}"
EXCLUDE="${EXCLUDE:-holygpu7c0705,holygpu8a25404,holygpu8a29104,holygpu7c[0522,0530],holygpu7c0928,holygpu8a[19101-19106,19201-19206,19301-19306,19401-19406,19501-19504,29406,29504-29506,31406,31504-31506],holygpu8a[19505-19506,19601-19606]}"
GPUS="${GPUS:-1}"
T_SMOKE="${T_SMOKE:-01:00:00}"; T_DIAG="${T_DIAG:-02:30:00}"; T_RM="${T_RM:-04:00:00}"; T_DPO="${T_DPO:-06:00:00}"; T_GRPO="${T_GRPO:-16:00:00}"; T_EVAL="${T_EVAL:-02:00:00}"
MEM="${MEM:-96G}"
mkdir -p runs; JOBS=runs/slurm_jobs.tsv; touch "$JOBS"

sb() {  # sb <name> <time> <partition> <dependency-or-empty> <export> <script> [account]
  local name="$1" time="$2" part="$3" dep="$4" exp="$5" script="$6" acct="${7:-$ACCOUNT}"
  local args=(--parsable --job-name="$name" --time="$time" --partition="$part" --account="$acct" --gres="gpu:$GPUS" --mem="$MEM"
              --constraint="$CONSTRAINT" --exclude="$EXCLUDE" --export="ALL,$exp")
  [ -n "$dep" ] && args+=(--dependency="$dep")
  [[ "$part" == *requeue* ]] && args+=(--requeue)
  local id; id=$(sbatch "${args[@]}" "$script")
  printf "%s\t%s\t%s\t%s\t%s\n" "$(date +%FT%T)" "$id" "$name" "$part" "$dep" >> "$JOBS"
  echo "$id"
}
lookup() { awk -F'\t' -v n="$1" '$3==n{id=$2} END{print id}' "$JOBS"; }

case "$STAGE" in
  smoke)
    sb smoke_dpo_1p5b "$T_SMOKE" "$WIDE_PARTITION" "" "SCRIPT=scripts/run_dpo_smoke_1p5b.sh" scripts/slurm/generic.sbatch
    for A in scalar shared separate_reward separate_cost; do
      sb "smoke_rm_$A" "$T_SMOKE" "$WIDE_PARTITION" "" "SCRIPT=scripts/run_rm_smoke.sh,ARGS=$A,RM_SMOKE_MODEL=Qwen/Qwen2.5-1.5B-Instruct" scripts/slurm/generic.sbatch
    done
    sb smoke_rm_init_diag "$T_SMOKE" "$WIDE_PARTITION" "" "SCRIPT=scripts/rm_init_diagnostic.py,PY=1" scripts/slurm/generic.sbatch
    sb eval_pilot_base "$T_SMOKE" "$WIDE_PARTITION" "" "SCRIPT=scripts/eval_pilot.py,PY=1,ARGS=--config configs/eval.yaml --n-prompts 40 --out runs/eval_pilot" scripts/slurm/generic.sbatch ;;
  smoke-rm)
    for A in scalar shared separate_reward separate_cost; do
      sb "smoke_rm_$A" "$T_SMOKE" "$WIDE_PARTITION" "" "SCRIPT=scripts/run_rm_smoke.sh,ARGS=$A,RM_SMOKE_MODEL=Qwen/Qwen2.5-1.5B-Instruct" scripts/slurm/generic.sbatch
    done ;;
  rm42)
    for A in scalar shared separate_reward separate_cost; do sb "rm_${A}_s42" "$T_RM" "$PARTITION" "" "ARCH=$A,SEED=42" scripts/slurm/rm_seed.sbatch; done ;;
  grpo-chain)
    SC=$(lookup rm_scalar_s42); SH=$(lookup rm_shared_s42); RR=$(lookup rm_separate_reward_s42); RC=$(lookup rm_separate_cost_s42)
    [ -n "$SC$SH$RR$RC" ] || { echo "run 'rm42' first"; exit 1; }
    CAL_SH=$(sb cal_shared "$T_SMOKE" "$WIDE_PARTITION" "afterok:$SH" "SCRIPT=scripts/run_grpo_calibrate.sh,ARGS=shared" scripts/slurm/generic.sbatch)
    CAL_SE=$(sb cal_separate "$T_SMOKE" "$WIDE_PARTITION" "afterok:$RR:$RC" "SCRIPT=scripts/run_grpo_calibrate.sh,ARGS=separate" scripts/slurm/generic.sbatch)
    REP_SH=$(sb repr_shared "$T_SMOKE" "$WIDE_PARTITION" "afterok:$CAL_SH" "PY=1,SCRIPT=-m saferl.grpo_calibrate,ARGS=--config configs/grpo_shared_constrained.yaml --stage representativeness" scripts/slurm/generic.sbatch)
    REP_SE=$(sb repr_separate "$T_SMOKE" "$WIDE_PARTITION" "afterok:$CAL_SE" "PY=1,SCRIPT=-m saferl.grpo_calibrate,ARGS=--config configs/grpo_separate_constrained.yaml --stage representativeness" scripts/slurm/generic.sbatch)
    sb cal_report "00:10:00" "$WIDE_PARTITION" "afterok:$REP_SH:$REP_SE" "PY=1,SCRIPT=scripts/grpo_calibration_report.py,NOGPU=1" scripts/slurm/generic.sbatch
    GP="$WIDE_PARTITION"
    SM_SC=$(sb smoke_grpo_scalar "$T_SMOKE" "$WIDE_PARTITION" "afterok:$SC" "SCRIPT=scripts/run_grpo_smoke_1p5b.sh,ARGS=scalar" scripts/slurm/generic.sbatch)
    SM_SH=$(sb smoke_grpo_shared "$T_SMOKE" "$WIDE_PARTITION" "afterok:$CAL_SH" "SCRIPT=scripts/run_grpo_smoke_1p5b.sh,ARGS=shared" scripts/slurm/generic.sbatch)
    SM_SE=$(sb smoke_grpo_separate "$T_SMOKE" "$WIDE_PARTITION" "afterok:$CAL_SE" "SCRIPT=scripts/run_grpo_smoke_1p5b.sh,ARGS=separate" scripts/slurm/generic.sbatch)
    for S in ${GRPO_SEEDS:-42 43 44}; do
      sb "grpo_scalar_s$S"   "$T_GRPO" "$GP" "afterok:$SM_SC" "ARM=scalar,SEED=$S"   scripts/slurm/grpo_seed.sbatch "$ACCOUNT_TRAIN"
      sb "grpo_shared_s$S"   "$T_GRPO" "$GP" "afterok:$SM_SH" "ARM=shared,SEED=$S"   scripts/slurm/grpo_seed.sbatch "$ACCOUNT_TRAIN"
      sb "grpo_separate_s$S" "$T_GRPO" "$GP" "afterok:$SM_SE" "ARM=separate,SEED=$S" scripts/slurm/grpo_seed.sbatch "$ACCOUNT_TRAIN"
    done ;;
  rest)
    for A in scalar shared separate_reward separate_cost; do for S in 43 44; do sb "rm_${A}_s$S" "$T_RM" "$PARTITION" "" "ARCH=$A,SEED=$S" scripts/slurm/rm_seed.sbatch; done; done
    for S in 42 43 44; do sb "dpo_s$S" "$T_DPO" "$PARTITION" "" "SEED=$S" scripts/slurm/dpo_seed.sbatch; done ;;
  eval)
    # Sharded evaluation (generate + score) gated on the corresponding FINAL training job (afterok) when that job id is known.
    # SEEDS="42" restricts to a subset of policy seeds (preliminary run); base (single response set) is always included.
    N="${N:-4}"; SEEDS="${SEEDS:-42 43 44}"; METHODS="${METHODS:-base dpo grpo_scalar grpo_shared grpo_separate}"
    EVAL_IDS=""
    if [[ " $METHODS " == *" base "* ]]; then EVAL_IDS="$(sb eval_base "$T_EVAL" "$WIDE_PARTITION" "" "METHOD=base,SEED=0,N=1,I=0" scripts/slurm/eval_shard.sbatch)"; fi
    for M in $METHODS; do [ "$M" = base ] && continue; for S in $SEEDS; do
      case "$M" in dpo) TJ=$(lookup "dpo_s$S");; grpo_scalar) TJ=$(lookup "grpo_scalar_s$S");; grpo_shared) TJ=$(lookup "grpo_shared_s$S");; grpo_separate) TJ=$(lookup "grpo_separate_s$S");; esac
      DEP=""; [ -n "$TJ" ] && DEP="afterok:$TJ"
      for I in $(seq 0 $((N-1))); do id=$(sb "eval_${M}_s${S}_$I" "$T_EVAL" "$WIDE_PARTITION" "$DEP" "METHOD=$M,SEED=$S,N=$N,I=$I" scripts/slurm/eval_shard.sbatch); EVAL_IDS="${EVAL_IDS:+$EVAL_IDS:}$id"; done
    done; done
    sb eval_merge "00:20:00" "$WIDE_PARTITION" "afterok:$EVAL_IDS" "PY=1,SCRIPT=-m saferl.eval_merge,ARGS=--config configs/eval.yaml --seeds $SEEDS" scripts/slurm/generic.sbatch ;;
  grpo-rerun42)
    # Corrected-sampler re-run of the seed-42 GRPO stage: archive the top-k=20 artifacts (nothing deleted), then
    # calibration -> representativeness -> report -> GRPO smokes -> GRPO seed-42 runs -> their eval shards -> merge.
    TS=$(date +%Y%m%d_%H%M); ARCH=runs/archive_topk20_$TS; mkdir -p "$ARCH/artifacts_grpo" "$ARCH/eval"
    for d in runs/grpo_scalar_qwen2.5-1.5b_seed42 runs/grpo_shared_constrained_qwen2.5-1.5b_seed42 runs/grpo_separate_constrained_qwen2.5-1.5b_seed42 \
             runs/grpo_smoke1p5b_scalar_seed42 runs/grpo_smoke1p5b_shared_seed42 runs/grpo_smoke1p5b_separate_seed42; do [ -e "$d" ] && mv "$d" "$ARCH/"; done
    for f in artifacts/grpo/calibration_generations.jsonl artifacts/grpo/calibration_generations.summary.json artifacts/grpo/train_sample_generations.jsonl \
             artifacts/grpo/train_sample_generations.summary.json artifacts/grpo/calibration_shared.json artifacts/grpo/calibration_shared.scores.jsonl \
             artifacts/grpo/calibration_separate.json artifacts/grpo/calibration_separate.scores.jsonl artifacts/grpo/calibration_shared_representativeness.json \
             artifacts/grpo/calibration_separate_representativeness.json artifacts/grpo/calibration_report.json; do [ -e "$f" ] && mv "$f" "$ARCH/artifacts_grpo/"; done
    for d in runs/eval/grpo_scalar runs/eval/grpo_shared runs/eval/grpo_separate; do [ -e "$d" ] && mv "$d" "$ARCH/eval/"; done
    cp runs/eval/summary.md "$ARCH/eval/summary_topk20_seed42.md" 2>/dev/null || true
    echo "archived top-k=20 artifacts under $ARCH"
    GRPO_SEEDS=42 bash "$0" grpo-chain
    SEEDS=42 METHODS="grpo_scalar grpo_shared grpo_separate" bash "$0" eval ;;
  eval-merge)
    SEEDS="${SEEDS:-42 43 44}"
    sb eval_merge "00:20:00" "$WIDE_PARTITION" "" "PY=1,SCRIPT=-m saferl.eval_merge,ARGS=--config configs/eval.yaml --seeds $SEEDS" scripts/slurm/generic.sbatch ;;
  grpo-gated42)
    # Corrected-sampler seed-42 GRPO: archive top-k=20 artifacts, regenerate calibration, then per arm
    # diagnostic (100 batches + gates, exit 2 on failure) -> continuation (resume from batch 100) -> eval shards -> merge.
    TS=$(date +%Y%m%d_%H%M); ARCHV=runs/archive_topk20_$TS; mkdir -p "$ARCHV/artifacts_grpo" "$ARCHV/eval"
    for d in runs/grpo_scalar_qwen2.5-1.5b_seed42 runs/grpo_shared_constrained_qwen2.5-1.5b_seed42 runs/grpo_separate_constrained_qwen2.5-1.5b_seed42 \
             runs/grpo_smoke1p5b_scalar_seed42 runs/grpo_smoke1p5b_shared_seed42 runs/grpo_smoke1p5b_separate_seed42; do [ -e "$d" ] && mv "$d" "$ARCHV/"; done
    for f in artifacts/grpo/calibration_generations.jsonl artifacts/grpo/calibration_generations.summary.json artifacts/grpo/train_sample_generations.jsonl \
             artifacts/grpo/train_sample_generations.summary.json artifacts/grpo/calibration_shared.json artifacts/grpo/calibration_shared.scores.jsonl \
             artifacts/grpo/calibration_separate.json artifacts/grpo/calibration_separate.scores.jsonl artifacts/grpo/calibration_shared_representativeness.json \
             artifacts/grpo/calibration_separate_representativeness.json artifacts/grpo/calibration_report.json; do [ -e "$f" ] && mv "$f" "$ARCHV/artifacts_grpo/"; done
    for d in runs/eval/grpo_scalar runs/eval/grpo_shared runs/eval/grpo_separate; do [ -e "$d" ] && mv "$d" "$ARCHV/eval/"; done
    cp runs/eval/summary.md "$ARCHV/eval/summary_topk20_seed42.md" 2>/dev/null || true
    echo "archived top-k=20 GRPO artifacts under $ARCHV (RM, DPO, base untouched)"
    CAL_SH=$(sb cal_shared "$T_SMOKE" "$WIDE_PARTITION" "" "SCRIPT=scripts/run_grpo_calibrate.sh,ARGS=shared" scripts/slurm/generic.sbatch)
    CAL_SE=$(sb cal_separate "$T_SMOKE" "$WIDE_PARTITION" "afterok:$CAL_SH" "SCRIPT=scripts/run_grpo_calibrate.sh,ARGS=separate" scripts/slurm/generic.sbatch)  # after shared: shares the generations file
    REP_SH=$(sb repr_shared "$T_SMOKE" "$WIDE_PARTITION" "afterok:$CAL_SH" "PY=1,SCRIPT=-m saferl.grpo_calibrate,ARGS=--config configs/grpo_shared_constrained.yaml --stage representativeness" scripts/slurm/generic.sbatch)
    REP_SE=$(sb repr_separate "$T_SMOKE" "$WIDE_PARTITION" "afterok:$CAL_SE:$REP_SH" "PY=1,SCRIPT=-m saferl.grpo_calibrate,ARGS=--config configs/grpo_separate_constrained.yaml --stage representativeness" scripts/slurm/generic.sbatch)
    sb cal_report "00:10:00" "$WIDE_PARTITION" "afterok:$REP_SH:$REP_SE" "PY=1,SCRIPT=scripts/grpo_calibration_report.py" scripts/slurm/generic.sbatch
    EVAL_IDS=""
    for ARM in scalar shared separate; do
      case "$ARM" in scalar) DEP="";; shared) DEP="afterok:$CAL_SH";; separate) DEP="afterok:$CAL_SE";; esac
      DIAG=$(sb "grpo_${ARM}_s42_diag100" "$T_DIAG" "$WIDE_PARTITION" "$DEP" "ARM=$ARM,SEED=42,EXTRA=--set stop_after_batches=100" scripts/slurm/grpo_seed.sbatch)
      CONT=$(sb "grpo_${ARM}_s42" "$T_GRPO" "$WIDE_PARTITION" "afterok:$DIAG" "ARM=$ARM,SEED=42" scripts/slurm/grpo_seed.sbatch "$ACCOUNT_TRAIN")
      for I in 0 1 2 3; do id=$(sb "eval_grpo_${ARM}_s42_$I" "$T_EVAL" "$WIDE_PARTITION" "afterok:$CONT" "METHOD=grpo_$ARM,SEED=42,N=4,I=$I" scripts/slurm/eval_shard.sbatch); EVAL_IDS="${EVAL_IDS:+$EVAL_IDS:}$id"; done
    done
    sb eval_merge_s42 "00:20:00" "$WIDE_PARTITION" "afterok:$EVAL_IDS" "PY=1,SCRIPT=-m saferl.eval_merge,ARGS=--config configs/eval.yaml --seeds 42" scripts/slurm/generic.sbatch ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "submitted; see $JOBS and: squeue -u \$USER -o '%.10i %.28j %.9P %.2t %.10M %.6D %R'"
