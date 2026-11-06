#!/usr/bin/env bash
# CPU-only job (no GPU allocation) for merging, bootstraps, selection and manifests. Partition/account set by submit_sixarm.sh
# (a FASRC CPU partition: Kempner GPU partitions must not run CPU-only work).
#SBATCH --job-name=sixarm_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm_%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/slurm/cluster_env.sh
export CUDA_VISIBLE_DEVICES=""
bash -c "${CMD:?CMD}"
