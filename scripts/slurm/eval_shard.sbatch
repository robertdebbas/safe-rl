#!/usr/bin/env bash
# One evaluation shard (generate + score) on one GPU.
#   sbatch --export=METHOD=dpo,SEED=42,N=4,I=0 scripts/slurm/eval_shard.sbatch
#   for I in 0 1 2 3; do sbatch --export=METHOD=dpo,SEED=42,N=4,I=$I scripts/slurm/eval_shard.sbatch; done
#SBATCH --job-name=eval_shard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm_eval_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/cluster_env.sh
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
bash scripts/run_eval_shard.sh "${METHOD:?METHOD}" "${SEED:?SEED}" "${N:-1}" "${I:-0}"
