#!/usr/bin/env bash
# Pending / running / completed status of the pipeline jobs recorded in runs/slurm_jobs.tsv (run on the cluster).
cd "$(dirname "$0")/../.."
echo "== queue =="; squeue -u "$USER" -o '%.10i %.28j %.12P %.2t %.10M %.12l %.4C %.8m %R' | head -60
echo "== recent accounting =="; sacct -u "$USER" -S "$(date -d '-3 days' +%F 2>/dev/null || date -v-3d +%F)" -o JobID%12,JobName%28,Partition%12,State%12,Elapsed%11,MaxRSS%10,AllocTRES%30 | grep -v "\.batch\|\.extern" | tail -60
