# Artifacts: where checkpoints, evaluation outputs and diagnostics live, and how to access them

Nothing large is tracked in Git. `artifacts/` (derived data, ~100 MB) and `runs/` (checkpoints, evaluation outputs, ~1 TB) are
ignored. Data is regenerated deterministically; models and outputs are read from the cluster.

## Live copy (FASRC Cannon / Kempner, read access needed)

```
/n/netscratch/calmon_lab/Lab/rdebbas/safe-rl/code/            the working copy (code + artifacts/ + runs/), group calmon_lab
```

| Path under `runs/` | Content | Size |
|---|---|---|
| `dpo_qwen2.5-1.5b_seed42/checkpoints/final` | DPO policy (bf16) | 3 GB |
| `rm_scalar_qwen2.5-1.5b_seed42/checkpoints/best`, `rm_shared_…`, `rm_separate_reward_…`, `rm_separate_cost_…` | the frozen reward/cost models (`model.safetensors` + `heads.pt` + `config.json`) and `eval/{val,test_*}_rows.jsonl` | ~3 GB each |
| `grpo_scalar_qwen2.5-1.5b_seed42`, `grpo_shared_constrained_…`, `grpo_separate_constrained_…` | final policies, `rollout_batches.jsonl` (per-batch training log), `monitor/`, `diagnostic/` (batch-100 samples, collapse probes), `resume/latest.pt` (18 GB) | 3 GB + logs |
| `eval/{base,dpo,grpo_scalar,grpo_shared,grpo_separate}/seed*/responses.jsonl, scores.jsonl` | every evaluation response and its Beaver scores; `eval/summary.{md,json,csv}` | small |
| `diagnose_shared/` | cross-scored responses, reports, plots, examples (see `docs/EXPERIMENTS.md` §3) | small |
| `archive_topk20_*`, `archive_jointnorm_*` | superseded runs (sampler leak; joint normalization) — not evidence | — |
| `sixarm_*`, `eval/sixarm_arm*` | the active six-arm study (do not touch while running) | — |
| `artifacts/grpo/calibration_*`, `artifacts/sixarm/` | calibration generations and thresholds, prompt manifests | small |

Hugging Face cache with the base model, the dataset and the Beaver evaluators: `/n/netscratch/calmon_lab/Lab/rdebbas/hf_cache`
(`export HF_HOME=…`). Python environment: `/n/netscratch/calmon_lab/Lab/rdebbas/saferl-venv` (`scripts/slurm/cluster_env.sh`).

**Access.** You need a FASRC account and read permission on the directory (group `calmon_lab`, or an ACL granted by the owner:
`setfacl -R -m u:<user>:rX /n/netscratch/calmon_lab/Lab/rdebbas/safe-rl`). Copy only what you need; do not write into this tree.
Without cluster access, ask for a tarball of `runs/rm_*/checkpoints/best` (4 × 3 GB) plus `runs/eval` and `runs/diagnose_shared`
(a few hundred MB); that is enough for every reward-model investigation in `docs/COLLABORATOR_GUIDE.md`.

## Regenerating data locally

```bash
python scripts/prepare_data.py --config configs/data.yaml          # artifacts/ (hashes must match artifacts/splits/manifest_hashes.json)
bash scripts/run_grpo_prepare.sh && bash scripts/run_rebel_prepare.sh && python scripts/sixarm_prepare.py   # prompt manifests
```

The dataset is CC BY-NC 4.0: keep derived splits private, do not commit them.

## Cluster etiquette (Kempner)

One GPU per job, ≤ 16 cores / 240–360 GB per GPU depending on partition, 2-day limit, no CPU-only jobs on Kempner GPU partitions
(use a FASRC CPU partition), a job idling a GPU for an hour is cancelled, 16 GPUs per user. Code, models and outputs produced on
Kempner fall under its open-science provisions (release by publication). Submit nothing against the active study's directories.
