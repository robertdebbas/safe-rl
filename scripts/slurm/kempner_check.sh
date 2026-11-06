#!/usr/bin/env bash
# Read-only Kempner preflight: what account / partitions / GPU types / limits this user actually has, and free disk. No job is submitted.
set -uo pipefail
echo "== user / groups";        id -nu; id -nG | tr ' ' '\n' | grep -i -E "kempner|lab" | tr '\n' ' '; echo
echo "== slurm associations";  sacctmgr -n -P show assoc user="$(id -nu)" format=account,partition,qos,maxjobs,maxsubmit,grptres 2>/dev/null
echo "== fairshare";           sshare -U -u "$(id -nu)" -o Account,RawShares,NormShares,RawUsage,EffectvUsage,FairShare 2>/dev/null
echo "== kempner partitions";  sinfo -p kempner,kempner_h100,kempner_h200,kempner_rtx,kempner_requeue -o "%P %a %D %T %G %m %c %l" 2>/dev/null | sort -u
echo "== kempner qos limits";  sacctmgr -n -P show qos format=name,maxtrespu,maxtrespa,maxjobspu,maxwall 2>/dev/null | grep -i kempner
echo "== idle GPUs by partition"
for p in kempner kempner_h100 kempner_h200 kempner_requeue; do echo -n "$p: "; sinfo -p "$p" -h -t idle -o "%D %G" 2>/dev/null | tr '\n' ';'; echo; done
echo "== disk";                df -h /n/netscratch 2>/dev/null | tail -1; du -sh "$(pwd)/runs" 2>/dev/null
echo "== python / torch";      source scripts/slurm/cluster_env.sh >/dev/null 2>&1 || true; python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())" 2>/dev/null
