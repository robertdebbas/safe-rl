# Constrained REBEL: algorithm, configuration and controls (seed-42 comparison)

Status: implemented and unit/CPU-tested; **no GPU job has been launched**. Paper: Gao et al., *REBEL: Reinforcement Learning via
Regressing Relative Rewards* (arXiv:2404.16767, Algorithm 1 / Eq. 1). Reference code: `ZhaolinGao/REBEL`, `src/tldr/rebel.py`.

## 1. Arms and controls

| Arm | Scorer(s) (frozen, RM seed 42, best-by-val) | Objective |
|---|---|---|
| A. Scalar REBEL | `rm_scalar` | maximize E[F~] |
| B. Shared constrained REBEL | `rm_shared` reward head + cost head | maximize E[R~] s.t. E[C~] <= d |
| C. Separate constrained REBEL | `rm_separate_reward`, `rm_separate_cost` | same |

Shared by all arms: Qwen2.5-1.5B-Instruct policy (full fine-tuning, fp32 master + bf16 autocast), policy seed 42, the SAME 9,054 training
prompts in the SAME seed-dependent order (`prompt_order(42)`, the GRPO rule), the same generation settings (T = 1, top_p = 1, **top_k = 0 pinned**,
repetition penalty 1, 512 new tokens, Qwen chat template, EOS = `<|im_end|>`/`<|endoftext|>`, truncation flagged), the same budget (one prompt epoch =
566 rollout batches of 16 x 8, 1,132 optimizer steps), the unchanged 1,960-prompt Beaver evaluation.

**Policy-training prompts (new).** `saferl.rebel_data` draws from the official train split (pinned revision) the prompts never used anywhere:
not in RM/DPO train (10k pairs / 9,054 prompts), val, either test split, nor ANY official test prompt (even invalid rows), nor the old GRPO
calibration/monitoring prompts. Pool after exclusion: 21,496 unique prompts (dataset-level numbers in `artifacts/rebel/manifest_info.json`).
Stratified by `prompt_source` to the old GRPO mix (Alpaca3-70B 63.2 % / Beavertails 21.9 % / WizardLM 14.9 %), data seed 42:
train 9,054, calibration 256, monitoring 256, pairwise disjoint; every hash intersection is checked and recorded (all 0).

## 2. Fixed calibration (`saferl.rebel_calibrate`)

One common set: 256 calibration prompts x 4 completions from pi_0 (seeds 5000-5003). For every arm and head S: mu_S, sigma_S (population std) over
the 1,024 responses, with prompt-cluster bootstrap SEs and quantiles. S~ = (S - mu_S) / sigma_S is frozen for the run (trainer checks scorer
fingerprints and the calibration's generation hash). Constrained arms: d = -0.5 standardized <=> d_raw = mu_C - 0.5 sigma_C; the file also records
alpha = 0 and alpha = 1 (d_raw = mu_C, mu_C - sigma_C) and attainability under pi_0 (fraction of responses <= d, fraction of prompts whose
best-of-4 is <= d). A non-finite or sigma < 1e-3 estimate raises. For reference, the old GRPO arms ended at about -1.0 (shared) and -2.2 (separate)
standardized cost units below the initial mean, so -0.5 is attainable under the learned cost models; matching d across arms does not match
Beaver safety.

## 3. Objective (exact)

Per rollout batch t, with pi_t the current weights:

* l(y) = SUM_k log pi(y_k | x, y_<k) over completion tokens (EOS included, prompt/padding masked; causal shift `logits[:, t] -> ids[:, t+1]`).
  Log-probs are the model's own (T = 1 sampling, so no temperature scaling question arises; the reference code divides logits by T = 0.7).
* Pairs: the 8 completions of a prompt -> 4 disjoint pairs by a permutation seeded by (seed, batch index, prompt position) only. 64 pairs/batch.
* Target z = (R~(y1) - R~(y2)) - lambda_t (C~(y1) - C~(y2)) (arm A: F~(y1) - F~(y2)); lambda_t fixed for the whole batch; z, scores, lambda,
  and l_t are detached constants.
* Loss (paper Eq. 1, 1/eta form): L = mean_pairs [ ( ((l_theta(y1) - l_t(y1)) - (l_theta(y2) - l_t(y2))) / eta - z )^2 ].
  Mean of squared residuals; no clipping, no advantages, no ratio surrogate, no whitening, no value function. Gradient flows only through
  l_theta. The **reference code uses the multiplied form** `(dl - eta * dr)^2` (gradient scale differs by eta^2); we follow the paper as requested.
* Optimization: 2 minibatches of 32 pairs, one AdamW step each, `inner_epochs = 1` (every pair once). The reference uses 4 inner epochs over the
  same rollout data; we keep 1 to match the GRPO update count (1,132 steps). Old log-probs l_t are computed once before any step and stored
  (detached); they are never recomputed with the updated policy.
* Dual (after the primal steps, once per batch): lambda <- max(0, lambda + 0.01 (C_hat - d)), C_hat = mean C~ over all 128 responses sampled
  from pi_t. Because the primal update uses lambda_t and the dual uses the pre-update rollouts, the cost estimate lags the updated policy by one
  iteration. No EMA, no log-lambda, no cap. Sign: cost above d -> lambda increases.
* KL regularization to pi_0: **off**. `kl_to_initial_coef > 0` would implement the paper's Eq. 22 reward shaping as an explicit separate option.
  The reference TL;DR runs used gamma = 0.05 plus a -1 reward for non-terminated responses; neither is active here.

Equivalence check (tested): exact minimization over all pairs on a finite action set recovers pi_{t+1} ∝ pi_t exp(eta [R~ - lambda C~]).

## 4. Proposed hyperparameters (not yet validated on GPU)

| Setting | Proposed | REBEL paper / code (TL;DR, Pythia + LoRA r=1024) | Why |
|---|---|---|---|
| eta | 1.0 | 1.0 (ablation best among 0.3/0.7/1/2) | paper default; standardized targets have unit scale |
| optimizer lr | 1e-6, cosine, 3 % warmup, AdamW (0.9, 0.999), wd 0, clip 1.0 | 3e-6, linear decay, AdamW defaults | same optimizer as our GRPO arms for comparability; full fine-tuning of 1.5B vs their LoRA |
| rollout | 16 prompts x 8 samples, 4 pairs/prompt | 2 samples/prompt, batch 512 | matches our GRPO budget and pairs every sample once |
| steps / batch | 2 x 32 pairs, 1 inner epoch | 4 inner epochs | matches GRPO's 2 steps per batch |
| dual | lambda_0 = 1, beta = 0.01, d = -0.5 | n/a | as specified |

Scale note: the regression residual is on sequence SUMS of log-probs (hundreds of tokens), so per-step gradient norms will be larger than
GRPO's per-token-mean surrogate; gradient clipping at 1.0 bounds the update and the 100-batch gate reports the regression fit (post-update
prediction/target correlation, loss at theta_t vs after the step, KL(pi_t || pi_{t+1})). If the fit does not improve after the step (corr ~ 0)
the lr is too small for eta; if KL(pi_t || pi_{t+1}) per token exceeds ~0.05 the step is too large. Either would be reported for a decision, not
changed silently. Dual stability: with standardized costs |C_hat - d| is O(1), so lambda moves <= ~0.01 per batch (<= ~5 over the epoch).

## 5. Logging, gates, artifacts

Per batch (`rollout_batches.jsonl`): loss at theta_t (= mean z^2), minibatch loss, post-update loss and pred/target correlation, target mean/std,
R~/C~ (or F~) mean/std and raw means, lambda before/after, unprojected lambda, violation, fraction of batches violating, KL(pi_t || pi_0) and
KL(pi_t || pi_{t+1}) (k3 estimator exp(r) - r - 1, per token and per sequence), gradient norms, lr, lengths, truncation/EOS rates, pair dR/dC
statistics, reward-length and cost-length Spearman, therapy-term density / refusal rate / longest sentence, non-finite flags, timings, memory.
Monitoring every 100 batches (256 prompts x 4, samples saved), generation dumps every 25 batches (32 prompts x 1 with scores), bf16 checkpoints
every 100 batches (`checkpoints/batch_00100`...; refused by the evaluation rule) and `checkpoints/final`, resume state every 25 batches
(verified after writing). Hard failures (exit 3, resume saved): NaN/Inf, log-prob > 0, non-finite gradient norm, lambda < 0 or non-finite,
unreadable resume file. Everything else (KL, lambda, length drift, truncation, tail mass, style) is reported, never gating. Gate run:
`--set stop_after_batches=100` -> diagnostic/batch_00100.json, exit 2 only on hard failure.

## 6. Matched GRPO (prepared, not launched)

`saferl.grpo_matched` subclasses the validated GRPO trainer: same REBEL prompts/order, standardized scorers, d = -0.5, additive projected dual.
Remaining algorithm differences vs REBEL: the policy update (group-normalized clipped surrogate vs squared regression) and GRPO's explicit
KL-to-pi_0 term (beta 0.02). `train_grpo.py` is unmodified (the only shared-code change is one added checkpoint-rule name in `eval_common`).

## 7. Tests (`tests/test_rebel.py`, 17 tests, CPU)

Loss at theta_t = mean z^2; swap symmetry; mean-of-squares and 1/eta gradient scaling; detached targets; tabular exponential-tilt recovery (LBFGS);
fixed disjoint reward-independent pairing; dual direction/projection/logging; standardized vs raw threshold equivalence; KL estimators; sequence
sums with causal shift / mask / EOS / padding vs a manual computation and padded-vs-unpadded equality; stored old log-probs untouched across
minibatches at lr > 0 and no scorer/reference gradients; manifest disjointness (forbidden prompt injected into the pool is excluded) and
stratification; config consistency (eta != lr, common keys, eval rule); tiny end-to-end run with resume reproducing an uninterrupted run,
dual-units check from the logs, gate stop/resume, scalar arm without dual, matched GRPO with the additive dual.
