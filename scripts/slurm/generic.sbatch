#!/usr/bin/env bash
# Generic one-GPU job: runs SCRIPT (bash script, or python when PY=1) with ARGS.  Used by submit_pipeline.sh.
#SBATCH --job-name=saferl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm_%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/cluster_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if [ "${PY:-0}" = "1" ]; then python3 ${SCRIPT:?SCRIPT} ${ARGS:-}; else bash ${SCRIPT:?SCRIPT} ${ARGS:-}; fi
