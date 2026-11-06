#!/usr/bin/env bash
# One arm, one policy seed, one GPU (resumable: resubmit the same job after preemption).
#   sbatch --export=ARM=scalar,SEED=42 scripts/slurm/grpo_seed.sbatch
#SBATCH --job-name=grpo_arm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm_grpo_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/cluster_env.sh
export WORLD_SIZE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
bash scripts/run_grpo.sh "${ARM:?ARM}" "${SEED:-42}" ${EXTRA:-}
