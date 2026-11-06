# Six-arm study: REBEL vs GRPO, scalar vs separate reward/cost, matched budgets (seed 42)

Status: implemented and CPU-tested; **no job submitted**. Every stage (calibration, smoke, hyperparameter validation, training,
evaluation) waits for explicit approval. Nothing below changes a validated GRPO setting.

## 1. Frozen configuration (configs/sixarm/*.yaml)

| Arm | Config | Trainer | Scorers (frozen, RM seed 42, best-by-val) | Constraint |
|---|---|---|---|---|
| 0 | arm0_rebel_separate_d1 | REBEL | separate R + separate C | d1 = mu_C − 0.5 sigma_C |
| 1 | arm1_rebel_separate_d0 | REBEL | separate R + separate C | d0 = mu_C |
| 2 | arm2_rebel_scalar | REBEL | scalar RM | none |
| 3 | arm3_grpo_separate_d1 | GRPO (unchanged) | separate R + separate C | d1 |
| 4 | arm4_grpo_separate_d0 | GRPO (unchanged) | separate R + separate C | d0 |
| 5 | arm5_grpo_scalar | GRPO (unchanged) | scalar RM | none |

Shared by all six: Qwen2.5-1.5B-Instruct, full fine-tuning (fp32 master, bf16 autocast), policy seed 42; `artifacts/sixarm` prompts;
200 rollout batches of 16 prompts × 8 samples = 25,600 responses; `max_rollout_batches: 200` with the optimizer schedule defined over the
full 566-batch epoch (1,132 steps, cosine, 3 % warmup) so a 200-batch run is the exact prefix of a full run and the saved resume state
continues it; checkpoints after batches 100 and 200 (`checkpoint_at_batches`); generation T = 1, top_p = 1, top_k = 0 pinned,
512 new tokens, official chat template; 256-prompt × 4 monitoring at start/100/200 and 32-prompt dumps every 25 batches; the same
AdamW settings (lr 1e-6 nominal, betas 0.9/0.999, eps 1e-8, wd 0, clip 1.0). Raw scores everywhere (no standardization).

**Prompts.** `artifacts/sixarm/{grpo,rebel}/train_prompts.jsonl` are the 9,054 unused official train prompts (disjoint by hash from
RM/DPO train and val, both test splits, every official test prompt and the old GRPO manifests). Both trainers apply the same
`prompt_order(9054, 42)`; the first 200 batches expose the same 3,200 prompts in the same order, listed in
`artifacts/sixarm/prompt_exposure.json` with zero intersections against every split and the 1,960 evaluation prompts.

**Optimizer steps.** GRPO: 2 minibatches of 8 groups → 2 steps per batch (unchanged). REBEL: 64 pairs → 2 minibatches of 32 → 2 steps per
batch, one inner epoch. Equal step counts (400 per arm) and equal prompt exposure.

**Dual (arms 0, 1, 3, 4).** GRPO's validated rule in raw cost units, identical hyperparameters in both trainers: EMA(0.9) of the
batch mean cost, mu ← mu + 0.01 · lambda · (C_ema − d), lambda = exp(mu), lambda_0 = 1, one update per batch after the primal steps.
This supersedes the additive projected dual proposed earlier for REBEL (kept in the code as an option, off). Differences to be aware
of: log-lambda can decay toward 0 but never reach it; the step is multiplicative in lambda; the EMA lags the batch cost. GRPO is
not modified.

**REBEL objective (arms 0–2).** Pairs: the 8 samples of a prompt → 4 disjoint pairs by a permutation seeded by (seed, batch, prompt
position), independent of scores. Target z = ΔR − lambda_t ΔC (arm 2: ΔF), constant for the batch; loss = mean over pairs of
((Δl_theta(y1) − Δl_theta(y2))/eta − z)^2 with l = SUM of completion-token log-probs (EOS included, prompt/pad masked), old log-probs
stored once before any step; eta = 1.0; no clipping, no ratio surrogate, no group normalization, no KL term. Tests: loss at
theta_t = mean z^2, swap symmetry, 1/eta gradient scaling, detached targets, tabular exponential-tilt recovery, sum-not-mean,
old log-probs fixed across minibatches, no scorer gradients.

## 2. Calibration (scripts/sixarm_calibration.py)

Reuses the validated pi_0 calibration of the separate cost model (256 prompts × 4, `artifacts/grpo/calibration_separate.json`, the
same artifact the seed-42 GRPO separate arm used) — the scorer fingerprints are copied into the two threshold files and checked by
both trainers at start. From the same responses re-scored locally: mu_C = −4.1723 (prompt-cluster bootstrap SE 0.085),
sigma_C = 1.8032 (SE 0.089); d0 = −4.1723, d1 = −5.0739 (SE ≈ 0.10). Feasibility of d1 under pi_0: 32 % of responses and 66 % of
prompts' best-of-4 are already at or below it; the seed-42 GRPO separate arm reached a mean cost of −8.1. d1 is attainable.
The thresholds are fixed once written and never recomputed.

## 2b. Reference-KL shaping for REBEL (audit, 2026-11-05)

GRPO's loss carries beta = 0.02 × the per-token-mean k3 KL to pi_0. REBEL's paper counterpart is reward shaping (Eq. 22):
r ← r − gamma (log pi_t(y|x) − log pi_0(y|x)). The paper sums the log-ratio over tokens. Measured on the seed-42 GRPO separate run
(per-token KL 0.29 at batch 100, 0.55 at the end; 220–325 tokens), the summed log-ratio is 60–180 nats, so gamma = 0.02 on the
SUM is a shaping term of 1.3–3.6 raw units — as large as the pair reward difference itself (|ΔR| ≈ 1.3 under pi_0) — whereas
GRPO's per-token beta term weighs ≈ 0.006–0.011 against advantages of ≈ 0.8. The same number is therefore a very different
regularizer in the two forms. Implemented as `kl_to_initial_coef` with `kl_shaping_units: token_mean | sequence_sum`
(`rebel_core.kl_shaping`). **Final formulation:** REBEL uses the token-mean form, r ← r − γ · mean_t(log pi_t − log pi_0), with γ = 0.02
if the pilot at the selected (lr, eta) passes every gate and lowers the achieved KL to pi_0, otherwise γ = 0; the sequence-sum form stays
available but is not used. Achieved per-token KL(pi ‖ pi_0) is logged every batch for all arms (`kl_initial` for REBEL, `kl_mean` for
GRPO) and at batches 0/100/200 on the monitoring prompts; it is reported per arm next to the main table. GRPO is unchanged.

### Reward/cost normalization: GRPO vs REBEL (documented difference)
GRPO's advantage is A = A_R − λ A_C with A_R and A_C each centred and divided by their **within-group** standard deviation (8 samples of
one prompt, eps 1e-6): the policy gradient is invariant to the scale and offset of R and C, every group contributes with unit variance,
and λ weighs two unit-scale quantities. REBEL regresses on the **raw** pair differences ΔR − λ ΔC in the reward/cost models' native units
(|ΔR| ≈ 1.3, |ΔC| ≈ 1.9 under pi_0); its step size is set jointly by eta (target scale) and the learning rate, which is what the pilots
calibrate, and λ weighs raw-unit quantities, so the same λ means a different reward/cost trade-off in the two families even though the
dual rule and its hyperparameters are identical. Neither is changed; both are reported in their own units, and the external evaluation
is the common scale.

## 3. Hyperparameter validation (REBEL only; prespecified; training/dev data only)

GRPO keeps its validated settings. REBEL's eta and learning rate are validated on `artifacts/sixarm_hpval`: the 640 prompts of
batches 400–439 of the same seed-42 order (never exposed in the 200-batch runs, disjoint from calibration/monitoring/held-out).
Scalar REBEL arm, 40 batches, constant learning rate (the question is the per-step behaviour at the nominal rate):

| run | lr | eta |
|---|---|---|
| 1–3 | 1e-6, 3e-6, 1e-5 | 1.0 |
| 4–5 | the lr chosen by the gate | 0.5, 2.0 |

**Data.** Pilots train on 640 fresh prompts drawn from the still-unused official pool (`artifacts/sixarm_hpval`), never in the main
manifest (so no resumption of a main run can reach them). Selection uses the 256 **monitoring prompts as the held-out development set**:
no pilot and no main run trains on them; they are sampled at the start and end of every pilot and scored in-job by the Beaver evaluators
(`scripts/sixarm_hpval_devscore.py`). The test set is never touched.
**Schedule.** Every pilot runs under the ACTUAL main-run schedule (`schedule_total_batches: 566`: cosine over 1,132 steps, 34 warmup steps),
so a pilot is exactly the first 40 batches of a main run with those hyperparameters — the selected values are thereby verified under
the training schedule, warmup included.
**Pilots (10).** REBEL: lr ∈ {1e-6, 3e-6, 1e-5} at eta 1, γ 0 (3); at the selected lr: eta ∈ {0.5, 2} (2), γ 0.02 token-mean (1), the
constrained check = arm 0's objective (1). GRPO (unchanged trainer): lr ∈ {3e-7, 1e-6, 3e-6} (3). The seed-42 GRPO runs (lr 1e-6, same
schedule shape, old prompts) are reused as the reference for the per-step-movement gate (ratio 1.0004, clip fraction 0.03–0.05 %) but
not as a pilot, since they used different prompts and no held-out dev scoring.

Logged per batch: regression loss at theta_t and after the step, prediction/target correlation, per-token KL(pi_t ‖ pi_{t+1}) (k3),
KL(pi_t ‖ pi_0), gradient norms, lengths, truncation, shaping magnitude vs pair signal. **Gates (scripts/sixarm_hpval_select.py,
proposed thresholds — approve or adjust before the runs):**
G1 fit (REBEL): post-step regression loss below the pre-step loss on ≥ 80 % of batches; G2 per-step movement: REBEL mean per-token
KL(pi_t ‖ pi_{t+1}) ≤ 1e-3 (ten times what the seed-42 GRPO runs implied: mean ratio 1.0004, clip fraction 0.03–0.05 %), GRPO mean
clipped-token fraction ≤ 5 %;
G3 truncation at the end ≤ 25 %; G4 per-token KL(pi_final ‖ pi_0) ≤ 1.0; G5 no exploitation: not (modeled reward rose and dev Beaver
helpfulness fell by > 0.5); G6 no harm drift: dev Beaver harmfulness rose by ≤ 0.5. (0.5 ≈ 1.5 standard errors of a 256-sample Beaver
mean.) Rule: largest lr passing every gate at eta = 1, gamma = 0 (none → smallest lr, flagged for review); keep eta = 1 unless
another eta passes every gate and raises the fit fraction by ≥ 0.10; adopt gamma = 0.02 only if its run passes every gate and lowers the
achieved KL to pi_0. The constrained check must pass G1–G4 and G6; it cannot change the selection. GRPO: the largest lr in
{3e-7, 1e-6, 3e-6} passing every gate (none → 1e-6, the previously validated value, flagged). The selected values are written to the
arm configs and the run manifest before the six runs start.

**Constraint feasibility, rechecked on the seed-42 GRPO separate trajectory** (trained with d0): the cost EMA fell below d0 at batch 15
and below d1 at batch 34, and stayed below d1 for 94 % of the run while lambda decayed 1 → 0.37 (batch 100) → 0.16 (batch 200). Both
thresholds are attainable; the stricter one binds for only ~35 batches. The d0-vs-d1 contrast (arms 0 vs 1, 3 vs 4) will therefore
mostly reflect the early-training lambda path, not a persistently active constraint. Reported, not changed.

**GRPO clipping under one inner epoch** (measured on the seed-42 runs): the first minibatch has ratio = 1 exactly; the second sees a
policy one step old, mean ratio 1.0004, clip fraction 0.03 % (shared) / 0.05 % (separate) of tokens, max 0.17 %. Clipping is
effectively inactive; GRPO's update is the unclipped surrogate with its KL(0.02) term.

## 4. Kempner deployment (scripts/slurm/submit_sixarm.sh)

Variables ACCOUNT / PARTITION / GRES / CPUS / MEM are verified first by `submit_sixarm.sh check` (associations, partitions, GPU
types, QoS limits, idle GPUs, disk); nothing Cannon-specific is assumed. **22 GPU jobs in total, CPU-only work never takes a GPU:**
pilot1 (6, parallel: 3 REBEL lr + 3 GRPO lr, each = training + in-job dev scoring; these are also the smokes) → `select` on the login
node → pilot2 (4, parallel) → `train` (6, parallel) → `eval` (6, parallel: arm k on all 1,960 prompts in one job, arm 0's job also
evaluates base) → one CPU job on a FASRC CPU partition (merge, 2,000-replicate standard and paired bootstraps, manifest). Defaults to confirm: account `kempner_kdbrantley_lab`
(the association was missing on 2026-11-01), partition `kempner_h100`, 1 GPU, 16 cores, 96 GB (within the 24-core / 360 GB per-GPU
cap), 8 h limit. Six training jobs run concurrently (≤ 16 GPUs per user). Per arm: ~200 × 40 s rollouts + 3 monitoring passes +
2 checkpoints ≈ 2.5–3 h (GRPO) and ≈ 3–3.5 h (REBEL, one extra no-grad pass per batch); peak GPU memory ≈ 40 GB (seed-42 GRPO peaked
at 38 GB with the same models). Evaluation: one job per arm over all 1,960 prompts (generation ≈ 1.2–1.8 h + Beaver scoring
≈ 0.3 h; arm 0's job adds base, ≈ +1.5 h) ≈ 12–14 GPU-hours. Training ≈ 18 GPU-hours. Pilots: 10 × ≈ 0.6 h (40 batches + two 256-sample
monitoring passes + Beaver scoring) ≈ 6 GPU-hours. Outputs: `runs/sixarm_arm<k>_*_seed42/` per arm; manifest
`runs/sixarm_run_manifest.json`.

## 5. Evaluation (configs/eval_sixarm.yaml, scripts/eval_pairs.py)

Unchanged protocol: the 1,960 prompts, per-prompt seeds shared by every method, T = 1 / top_p = 1 / 512 tokens, Beaver unified reward
(helpfulness ↑) and cost (harmfulness ↓), 2,000 prompt-level bootstrap replicates. Every arm at `checkpoints/batch_00200`
(rule `after_batches:200`, which refuses any other checkpoint). Primary deliverable: the seven-row table (base + six arms) from
`eval_merge --methods ...`; paired prompt-level bootstrap CIs for 0 vs 3, 1 vs 4, 2 vs 5, 0 vs 1, 3 vs 4 from `eval_pairs.py`.
Reported separately per arm: learned-cost mean and violation (C − d), lambda trajectory, KL to pi_0, response length, truncation,
measured GPU-hours. One policy seed: all intervals quantify evaluation-prompt uncertainty only, not training-seed variation.

## 6. Comparability notes

* GRPO contains an explicit KL(0.02) term to pi_0; REBEL has none (its proximity to pi_t is the regression's 1/eta term). This is
  an algorithm difference, not a tuning difference, and is left as is.
* GRPO's advantage is group-normalized (scale-free); REBEL's target is in raw reward/cost units, so eta sets the effective step. The
  validation stage addresses this; learning rates may legitimately differ between the two families.
* Lengths: the seed-42 GRPO arms drifted to the 512-token cap; nothing here penalizes truncation, for either family.
* The constrained arms share one calibration (one cost model), so d0/d1 are comparable across arms by construction; they are not
  statements about Beaver harm.
