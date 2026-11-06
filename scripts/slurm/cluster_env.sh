#!/usr/bin/env bash
# Sourced by every sbatch script on the cluster. Edit ONCE for your environment (module names / conda env).
# Nothing here changes scientific settings.
module purge 2>/dev/null || true
module load ${SAFERL_PYTHON_MODULE:-python/3.10.13-fasrc01} 2>/dev/null || true
module load ${SAFERL_CUDA_MODULE:-cuda/12.4.1-fasrc01} 2>/dev/null || true
if [ -n "${SAFERL_CONDA_ENV:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$SAFERL_CONDA_ENV"
elif [ -f "${SAFERL_VENV:-/n/netscratch/calmon_lab/Lab/rdebbas/saferl-venv}/bin/activate" ]; then
  source "${SAFERL_VENV:-/n/netscratch/calmon_lab/Lab/rdebbas/saferl-venv}/bin/activate"
fi
export PYTHONPATH="${SLURM_SUBMIT_DIR:-$PWD}/src"
export HF_HOME="${HF_HOME:-/n/netscratch/calmon_lab/Lab/rdebbas/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export WORLD_SIZE=1
