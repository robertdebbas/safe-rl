#!/usr/bin/env bash
# One REBEL arm, one policy seed, one GPU (resumable: resubmit the same job after preemption).
#   sbatch --export=ARM=scalar,SEED=42 scripts/slurm/rebel_seed.sbatch        (EXTRA="--set stop_after_batches=100" for the gate run)
#   MATCHED=1 runs the prepared matched-GRPO comparison instead (scripts/run_grpo_matched.sh).
#SBATCH --job-name=rebel_arm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm_rebel_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/cluster_env.sh
export WORLD_SIZE=1
if [ "${MATCHED:-0}" = "1" ]; then
  bash scripts/run_grpo_matched.sh "${ARM:?ARM}" "${SEED:-42}" ${EXTRA:-}
else
  bash scripts/run_rebel.sh "${ARM:?ARM}" "${SEED:-42}" ${EXTRA:-}
fi
