#!/usr/bin/env bash
# One arm of the six-arm study on ONE GPU (resumable: resubmitting the same job continues from runs/<run>/resume/latest.pt).
#   sbatch --export=ALL,ARM_CONFIG=configs/sixarm/arm0_rebel_separate_d1.yaml scripts/slurm/sixarm_train.sbatch
# Kempner limits respected: 1 GPU, <= 16 cores, well under the per-GPU memory cap, < 2 days. Resource lines are overridden by
# submit_sixarm.sh (partition / account / gres / time) so this file carries no Cannon-specific names.
#SBATCH --job-name=sixarm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm_sixarm_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/cluster_env.sh
export WORLD_SIZE=1
CFG="${ARM_CONFIG:?ARM_CONFIG}"
case "$CFG" in *rebel*|*hpval*) MOD=saferl.train_rebel;; *grpo*) MOD=saferl.train_grpo;; *) echo "cannot infer trainer from $CFG"; exit 1;; esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -m "$MOD" --config "$CFG" --seed "${SEED:-42}" ${EXTRA:-}
