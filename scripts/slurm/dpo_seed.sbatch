#!/usr/bin/env bash
# One scientific seed on one GPU.   sbatch --export=SEED=42 scripts/slurm/dpo_seed.sbatch
# Adjust partition/account/modules to your cluster before use.
#SBATCH --job-name=dpo_arm1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/cluster_env.sh
export WORLD_SIZE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
bash scripts/run_dpo.sh "${SEED:-42}"
