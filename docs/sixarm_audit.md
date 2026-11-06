# Six-arm study: final pre-launch audit (2026-11-05, revised after the final adjustments)

Companion to `docs/sixarm_spec.md`. Status: code complete, full test suite passing, nothing submitted. Every stage requires approval.

## A. Implementation

| Component | Where | What it does |
|---|---|---|
| Main prompts | `scripts/sixarm_prepare.py` → `artifacts/sixarm/{grpo,rebel}/*.jsonl`, `prompt_exposure.json` | 9,054 unused official train prompts, hash-disjoint from RM/DPO train+val, both test splits, all official test prompts, old GRPO manifests; both trainers apply `prompt_order(9054, 42)`; the first 200 batches = the same 3,200 prompts in the same order (listed; every intersection 0); full epoch 566 batches |
| Pilot prompts | `artifacts/sixarm_hpval/` | 640 FRESH prompts from the still-unused pool (stratified to the main mix): never in the main manifest, calibration, monitoring, any split or the evaluation set — no main-run resumption can reach them |
| Development set | the 256 monitoring prompts | sampled (not trained on) by pilots and main runs; Beaver-scored in-job for every pilot (`sixarm_hpval_devscore.py`); the 1,960 test prompts are never touched before the final evaluation |
| Thresholds | `scripts/sixarm_calibration.py` → `artifacts/sixarm/calibration_separate_{d0,d1}.json` | from the validated 256×4 pi_0 calibration of the separate cost model; d0 = mu_C, d1 = mu_C − 0.5 sigma_C (raw units); scorer fingerprints copied and verified by both trainers; bootstrap SEs; feasibility |
| GRPO arms 3–5 | `saferl.train_grpo` (objective unchanged) | group-normalized A_R − λA_C, clip 0.2, per-token k3 KL β 0.02, 2 minibatches of 8 groups; log-λ dual (EMA 0.9, step 0.01, λ0 1). Additions, all opt-in and default-off: `checkpoint_at_batches`, `schedule_total_batches`, `monitor_save_samples` |
| REBEL arms 0–2 | `saferl.train_rebel`, `saferl.rebel_core` | fixed reward-independent pairing; z = ΔR − λ_t ΔC (raw; arm 2: ΔF); loss = mean over pairs of ((Δl(y1) − Δl(y2))/η − z)², l = sum of completion-token log-probs; stored old log-probs; 2 minibatches of 32 pairs, 1 epoch; same dual rule/hyperparameters as GRPO; reference-KL shaping token-mean form, γ ∈ {0, 0.02} decided by the pilots |
| Generation / scoring / EOS | `grpo_rollout.generate_completions`, `grpo_scoring.FrozenScorers` (both trainers) | T 1, top_p 1, top_k 0 pinned, 512 tokens; stop on `<|im_end|>`/`<|endoftext|>`; capped responses kept and flagged; identical scorer encoding; truncation logged per batch everywhere; no length penalty (tested) |
| Monitoring, resume, checkpoints | both trainers | 256×4 monitoring at 0/100/200 (REBEL saves samples; REBEL dumps 32×1 every 25); resume every 25 batches (policy, optimizer, scheduler, cursor, λ, EMA, RNG) with a test that a cut-and-resumed run reproduces every logged quantity; checkpoints at batches 100 and 200; LR schedule over the full 566-batch epoch |
| Evaluation | `configs/eval_sixarm.yaml`, `saferl.eval_*`, `scripts/eval_pairs.py` | unchanged protocol (1,960 prompts, shared per-prompt seeds, Beaver evaluators, 2,000 replicates); rule `after_batches:200`; paired bootstraps for 0v3, 1v4, 2v5, 0v1, 3v4; one GPU job per arm, CPU-only merge |
| Manifest | `scripts/sixarm_manifest.py` | hashes of code, configs, artifacts, thresholds, selected hyperparameters, library versions |

## B. Hyperparameters (resolved)

| Setting | REBEL arms 0–2 | GRPO arms 3–5 |
|---|---|---|
| Policy / init | Qwen2.5-1.5B-Instruct, full FT, fp32 master + bf16 autocast, grad checkpointing, seed 42 | same |
| Budget | 200 batches × 16 × 8 = 25,600 responses; 400 optimizer steps | same |
| Optimizer | AdamW, lr **selected by pilots** from {1e-6, 3e-6, 1e-5}, β (0.9, 0.999), ε 1e-8, wd 0, clip 1.0, cosine over 1,132 steps, 3 % warmup | AdamW, lr **selected by pilots** from {3e-7, 1e-6, 3e-6}, otherwise identical |
| Objective | squared regression, η selected from {0.5, 1, 2} (default 1), raw scores | clipped surrogate ε 0.2, group-normalized advantages (eps 1e-6), β 0.02 per-token k3 KL to pi_0 |
| Reference KL | γ ∈ {0, 0.02} token-mean shaping, decided by the pilot | β 0.02 in the loss |
| Dual (arms 0,1 / 3,4) | log-λ, λ0 1, step 0.01, EMA 0.9, raw d; d0 = −4.172, d1 = −5.074 | same |
| Generation | T 1, top_p 1, top_k 0, 512 new tokens, prompt ≤ 512 | same |
| Micro-batch | 8 trajectories | 8 trajectories |

## C. Jobs (scripts/slurm/submit_sixarm.sh): 22 GPU jobs, 1 CPU job

| Stage | Job names | Jobs | Each (limit) | After |
|---|---|---|---|---|
| check, calibrate, select | login node (read-only / seconds) | 0 | — | — |
| pilot1 | `pilot_rebel_lr1e-6`, `pilot_rebel_lr3e-6`, `pilot_rebel_lr1e-5`, `pilot_grpo_lr3e-7`, `pilot_grpo_lr1e-6`, `pilot_grpo_lr3e-6` | 6 GPU, parallel | ~35 min (2.5 h): 40 batches + 2 monitoring passes + Beaver dev scoring | calibrate |
| pilot2 | `pilot_rebel_lr<LR>_eta0.5`, `pilot_rebel_lr<LR>_eta2.0`, `pilot_rebel_lr<LR>_g0.02`, `pilot_rebel_sep_d1` | 4 GPU, parallel | ~35 min | select (login node) |
| train | `sixarm_arm0_rebel_separate_d1` … `sixarm_arm5_grpo_scalar` | 6 GPU, parallel | ~3 h (8 h) | selected values written to the REBEL and GRPO arm configs |
| eval | `eval_arm0` (also base) … `eval_arm5` → `eval_merge_pairs_manifest` | 6 GPU parallel + 1 CPU | ~1.5–2 h per arm, arm 0 ~3.5 h (5 h); CPU ~5 min | train |

GPU-hours ≈ 6 (pilots) + 18 (training) + 13 (evaluation) ≈ 37.

## D. GPU type

Peak training memory 38 GB on A100-80 with these models; the H100 (80 GB) and H200 (141 GB) both have ample headroom, 40 GB cards do not.
Generation dominates wall-clock and scales with memory bandwidth, so an H200 is ≈ 20–30 % faster per job but usually scarcer.
Recommendation: `kempner_h100` for every stage (6 parallel jobs sit under the 16-GPU user cap), falling back to `kempner_h200` for a stage
where the check shows idle H200s and no idle H100s. Per-GPU caps: H100 24 cores / 360 GB, H200 16 cores / 360 GB; the jobs request
16 cores / 96 GB. CPU-only work (merge, bootstraps, selection, manifests) runs on a FASRC CPU partition (`shared` by default), never on a
Kempner GPU partition. Avoid `kempner_requeue` for the training arms.

## E. Before launch

1. `submit_sixarm.sh check`: Kempner association (`kempner_kdbrantley_lab` was missing on 2026-11-01), partitions, idle GPUs, disk.
2. Approve the six gate thresholds (spec §3) — or adjust them before pilot1.
3. After pilot1 → `select` → pilot2 → `select`: write the selected REBEL lr/η/γ and the GRPO lr into the arm configs (one line each),
   rerun `calibrate` (regenerates the manifest), then `train`.
4. Standing caveats: GRPO's explicit KL term vs REBEL's optional shaping; group-normalized vs raw-unit targets (spec §2b); d1 binds only
   early on the seed-42 trajectory; no truncation penalty; one policy seed (intervals = evaluation-prompt uncertainty only).
