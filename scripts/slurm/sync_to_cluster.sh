#!/usr/bin/env bash
# rsync the repository + fixed artifacts (not runs/, not smoke/ref caches) to the cluster through the SSH ControlMaster socket.
#   bash scripts/slurm/sync_to_cluster.sh [host=cannon] [remote_dir=~/safe-rl/code]
set -euo pipefail
HOST="${1:-cannon}"; REMOTE="${2:-~/safe-rl/code}"
cd "$(dirname "$0")/../.."
ssh -o BatchMode=yes "$HOST" "mkdir -p $REMOTE"
# Code (src/scripts/configs/tests) is mirrored with --delete; artifacts/ are copied WITHOUT --delete so files generated on the
# cluster (calibration_*.json, *_generations.jsonl, GPU diagnostics) are never removed by a sync.
rsync -az --delete --exclude '.git/' --exclude '__pycache__/' --exclude '.pytest_cache/' -e "ssh -o BatchMode=yes" \
      src scripts configs tests pyproject.toml README.md .gitignore "$HOST:$REMOTE/"
rsync -az --exclude 'smoke/' --exclude 'ref_logprobs/' --exclude 'grpo/calibration*' --exclude 'grpo/*generations*' \
      --exclude 'rm_init_diagnostic.*' -e "ssh -o BatchMode=yes" artifacts/ "$HOST:$REMOTE/artifacts/"
echo "synced to $HOST:$REMOTE"
