# Safe RL arms — DPO (Arm 1), reward/cost models and (constrained) GRPO policy training (Arms 2–4)

Research question: *given separately labeled helpfulness and harmlessness preferences, how should we
represent these objectives and use them to train a policy?*  This repository currently implements only
**Arm 1**: vanilla sigmoid DPO that consumes both preference dimensions with equal weight and without
telling the policy which objective a comparison came from. Later arms (scalar RM + RL, shared-backbone
two-head RM + constrained RL, two independent RMs + constrained RL) will reuse the exact same data splits,
policy initialization, evaluation sets and seeds.

## Layout

```
configs/
  data.yaml          data preparation (official PKU-SafeRLHF, pinned revision, data_seed=42, 10k/2k/2k/2k)
  data_smoke.yaml    tiny deterministic subset for engineering smoke tests (artifacts/smoke)
  dpo.yaml           scientific DPO configuration (Qwen2.5-1.5B-Instruct, full fine-tuning)
  dpo_smoke.yaml     smoke training configuration (Qwen2.5-0.5B-Instruct, 4 steps; engineering only)
  rm_common.yaml     common optimization hyperparameters for ALL score-model architectures
  rm_scalar.yaml, rm_shared.yaml, rm_separate_reward.yaml, rm_separate_cost.yaml   arch = ... (inherit rm_common)
  rm_smoke.yaml      score-model smoke (0.5B; --set arch=...)
src/saferl/
  data.py            official loading, pair records, validity filtering, leakage removal, stratified sampling, DPO rows, manifests
  audit.py           token-length audit, threshold selection, distributions, Markdown report
  prepare.py         end-to-end data preparation (writes artifacts/)
  formatting.py      Qwen chat formatting, tokenization, prompt masking, collation, leakage assertion
  dpo_loss.py        masked sequence log-probs + sigmoid DPO loss
  modeling.py        model loading, bf16 autocast, batched log-probs, reference log-prob cache, checkpoint saving
  metrics.py         subgroup metrics (objective x pair type) + conflict diagnostic
  train_dpo.py       trainer (fp32 master weights, bf16 autocast, cosine schedule, validation-based checkpoint selection)
  eval_dpo.py        evaluation of a checkpoint on val / test_representative / test_balanced
  aggregate.py       mean ± std across seeds
  rm_data.py         reward/cost rows from the SAME split manifests (+ PKU pointwise safety labels), architecture table
  rm_models.py       ScoreModel: Qwen backbone (no LM head) + linear scalar heads; score_reward/score_cost/score_scalar
  rm_loss.py         Bradley-Terry loss, cost ranking + pointwise classification loss
  rm_metrics.py      RM metrics per objective x pair type (pair- and prompt-averaged)
  train_rm.py        trainer for Architectures A (scalar), B (shared two-head), C (separate_reward / separate_cost)
  eval_rm.py         evaluation of a score-model checkpoint
  aggregate_rm.py    mean ± std across seeds for score models
  grpo_data.py       GRPO prompt manifests (9,054 unique train prompts; 256 calibration + 256 monitoring from val), prompt order, batches
  grpo_core.py       group advantages, Z = R - lambda C, dual update (log-lambda), clipped objective + reference KL (pure math)
  grpo_rollout.py    HF generate rollouts (pi_old = current weights), trajectory collation, per-token log-probs
  grpo_scoring.py    frozen scorers from ScoreModel checkpoints (scalar | shared | separate), raw unanchored scores
  grpo_calibrate.py  pi_0 calibration generations (256 x 4, fixed seeds) and d_<arm> = mean cost under the arm's cost model
  train_grpo.py      (constrained) GRPO trainer: one prompt epoch, monitoring, resume, final checkpoint, budget logging
  eval_common.py     evaluation manifest (test_representative deduplicated: 1,960 prompts), per-prompt seeds, fixed batches, shards
  eval_generate.py   per-prompt-seeded sampling (inverse-CDF with one CPU generator per prompt) for one method/seed/shard
  eval_score.py      frozen PKU Beaver unified reward/cost evaluators (LlamaForScore loader), fixed-batch scoring
  eval_merge.py      shard verification, concatenation, seed means, seed-stratified bootstrap CIs, final table
scripts/
  prepare_data.py    python scripts/prepare_data.py --config configs/data.yaml
  run_dpo.sh         one scientific seed: train + eval
  run_dpo_all_seeds.sh   seeds 42, 43, 44 sequentially + aggregate
  run_dpo_smoke.sh   engineering smoke test (0.5B; laptop friendly)
  run_dpo_smoke_1p5b.sh  engineering smoke test with the scientific 1.5B model (GPU; run before full seeds)
  aggregate.sh       aggregate the three seeds
  slurm/dpo_seed.sbatch  one seed on one GPU under SLURM
  run_rm.sh <arch> <seed>   one score-model architecture + seed: train + eval
  run_rm_smoke.sh    engineering smoke for all four score-model trainings (0.5B)
  aggregate_rm.sh <arch>    aggregate seeds 42/43/44 of one architecture
  slurm/rm_seed.sbatch      sbatch --export=ARCH=<arch>,SEED=<seed>
  run_grpo_prepare.sh       build GRPO prompt manifests
  run_grpo_calibrate.sh <shared|separate>   pi_0 calibration generations + d_<arm>
  run_grpo.sh <arm> <seed>  one policy seed (resumable), run_grpo_all_seeds.sh <arm>, run_grpo_smoke.sh (0.5B)
  slurm/grpo_seed.sbatch    sbatch --export=ARM=<scalar|shared|separate>,SEED=<seed>
  run_eval_prepare.sh       evaluation prompt manifest
  run_eval_shard.sh <method> <seed> <N> <i>   generate + score one shard (single GPU); run_eval_merge.sh -> final table
  run_eval_smoke.sh         pipeline smoke with tiny synthetic evaluators (1 vs 2 shards identical); make_tiny_evaluators.py
  slurm/eval_shard.sbatch   sbatch --export=METHOD=<m>,SEED=<s>,N=<shards>,I=<shard>
tests/               pytest suite (data, DPO rows, tokenization, loss, artifacts, tiny end-to-end training)
artifacts/           generated by prepare_data.py (gitignored): splits, DPO rows, audit, length limits, ref log-prob cache
runs/                generated by training/eval (gitignored)
```

## Setup

```
python3 -m pip install -e ".[dev]"      # or: export PYTHONPATH=src
```
Python >= 3.9, torch >= 2.3, transformers >= 4.45, datasets >= 4. A Hugging Face token is needed only
if your Hub access requires one (the dataset is public, CC-BY-NC-4.0).

## Data design (see `artifacts/data_audit.md` after running the preparation)

* Source: `PKU-Alignment/PKU-SafeRLHF`, config `default` (union of the Alpaca-7B / Alpaca2-7B / Alpaca3-8B
  files = the full official dataset), pinned to revision `9421ffafec3fa40a1f1a7d567b4d525079477ecb`.
* `better_response_id` = helpfulness preference, `safer_response_id` = harmlessness preference;
  agreement = equal ids, conflict = different ids. Counts are recomputed from the official data.
* Validity: non-empty prompt / response_0 / response_1 and ids in {0, 1}; per-reason counts are logged.
* Prompt leakage: prompts are normalized (strip + collapse whitespace) only for comparison. **The official
  train and test splits share 6,141 normalized prompts**; every official-train pair whose prompt occurs in
  the official test split is dropped from the train/val pool, so the official test split stays intact and
  all four splits are prompt-disjoint.
* Token-length audit with the Qwen2.5 tokenizer and the real chat formatting; the smallest candidate
  thresholds covering >= 99% are selected (512 / 1024) and the (4) over-length underlying pairs are
  excluded before sampling rather than truncated.
* Sampling (data_seed=42, deterministic, order-independent): proportional stratified sampling within each
  pair type over strata (prompt_source x max severity x safety pattern). Order: train (5k/5k) ->
  val (1k/1k, prompts disjoint from train) -> test_balanced (1k/1k) -> test_representative (2000 at the
  natural agreement/conflict ratio of the eligible official test pool, prompts disjoint from test_balanced).
  A prompt used by one split is owned by it; unused pairs of that prompt go nowhere.
* Every underlying pair -> exactly two DPO rows (helpfulness: chosen = better_response_id; harmlessness:
  chosen = safer_response_id). Conflict pairs therefore produce two rows with opposite orderings; nothing is
  deduplicated. `objective` is metadata only and never enters the model input (tested).
* Manifests `artifacts/splits/*.manifest.jsonl` hold pair ids (sha256 of official split/index/texts),
  official indices, official response hashes and labels; `*.pairs.jsonl` hold the full text;
  `artifacts/dpo/*.jsonl` hold the DPO rows. `manifest_hashes.json` records sha256 of all files.
  Later arms must consume these files.

## Model / training design

* Policy pi_0 and frozen reference pi_ref are the same checkpoint `Qwen/Qwen2.5-1.5B-Instruct`;
  full-parameter fine-tuning (no LoRA).
* Formatting: official Qwen2.5 chat template, single user turn, no custom system prompt (the template itself
  inserts Qwen's default system line; see `artifacts/serialized_examples.txt`). Completion = response + `<|im_end|>`.
  Prompt tokens are context only (loss_mask = 0); the DPO log-prob is sum over response tokens of
  log pi(y_t | x, y_<t).
* Loss: sigmoid DPO, beta = 0.1, label_smoothing = 0, per-row equal weighting (H and S rows are equal in
  number, so the epoch-level objective is 0.5 L_H + 0.5 L_S). Rows are shuffled independently each epoch.
* Optimization: AdamW (0.9, 0.999, eps 1e-8, wd 0), lr 5e-7, cosine with 3% warmup, grad clip 1.0,
  global batch 64 rows = per_device 4 x world_size 1 x accumulation 16 (the trainer refuses inexact
  factorizations), max 2 epochs, bf16 autocast with fp32 master weights and fp32 AdamW states,
  gradient checkpointing.
* Reference log-probs are computed once with the model at initialization and cached in
  `artifacts/ref_logprobs/` (keyed by model, split hash, lengths, device type, precision); all seeds share it.
* Validation every 100 optimizer steps and at the end of each epoch. `checkpoints/best` = lowest
  0.5 * helpfulness_val_loss + 0.5 * harmlessness_val_loss (diagnostic; used by `eval_dpo` and the DPO
  aggregate), `checkpoints/final` = the end-of-epoch-2 checkpoint (the PRIMARY checkpoint for the common
  external comparison, symmetric with the GRPO arms' final one-epoch checkpoints). Kept: best, last/final,
  epoch ends (bf16 safetensors). Test data is never used for selection.
* Evaluation reports, for overall / objective / pair type / objective x pair type: loss, strict preference
  accuracy P(margin > 0), tie-adjusted accuracy (zero margin counts 0.5), the fractions of positive / zero /
  negative margins, mean implicit reward margin, chosen/rejected log-probs and counts (rows and prompts).
  Every statistic is given **pair-averaged** (each DPO row weighs 1) and **prompt-averaged** (rows averaged
  within a prompt, then prompts weighed equally, so prompts with many response pairs do not dominate).
  For every conflict pair it also reports whether the policy favors the helpfulness-preferred or the
  safety-preferred response (counts, fractions, ties, and prompt-averaged fractions).
  Checkpoint selection uses the pair-averaged combined validation loss.

## Commands

```
# 1. data (idempotent, deterministic; ~2 min CPU)
python scripts/prepare_data.py --config configs/data.yaml
python -m pytest tests                      # add -m "not slow" to skip the regeneration check

# 2. engineering smoke tests (4 steps on a tiny subset)
bash scripts/run_dpo_smoke.sh            # 0.5B model; laptop MPS/CPU or GPU
bash scripts/run_dpo_smoke_1p5b.sh       # scientific 1.5B model; GPU; run this before any full seed

# 3. one scientific seed on one GPU (train + eval)
bash scripts/run_dpo.sh 42
sbatch --export=SEED=42 scripts/slurm/dpo_seed.sbatch        # SLURM variant

# 4. all three seeds, then aggregate
bash scripts/run_dpo_all_seeds.sh                            # or three sbatch jobs then:
bash scripts/aggregate.sh                                    # -> runs/aggregate/summary.{json,md}
```

Per-run outputs live in `runs/<run_name>_seed<seed>/`: `config_resolved.json`, `metrics.jsonl` (train
steps + validation points), `val_history.json`, `train_summary.{json,md}`, `checkpoints/{best,step_*}`,
`eval/{val,test_representative,test_balanced}.json`, `eval/*_rows.jsonl`, `eval/eval_summary.md`.

## Reward / cost models (Arms 2–4 components; no RL yet)

All architectures consume exactly the DPO manifests (`artifacts/splits/*.pairs.jsonl`): the same
10k train / 2k val / 2k + 2k test pairs, each giving one helpfulness comparison (`<pair_id>:H`,
chosen = `better_response_id`) and one safety comparison (`<pair_id>:S`, chosen = `safer_response_id`
= y_safe, rejected = y_harmful) plus PKU's pointwise `is_response_X_safe` labels.

* Score = `w^T h_last + b` at the hidden state (after the final norm) of the last non-padding token
  of prompt + response + `<|im_end|>`, i.e. the appended EOS; normal(0, initializer_range) init, zero
  bias, fp32 heads. Prompt tokens are context only; the loss is only on the scalar.
* **A `scalar`**: one head F on the union of H and S rows, `L = -log sigmoid(F(y+) - F(y-))`,
  equal row counts so the epoch objective is 0.5 L_H + 0.5 L_S. No pointwise labels, no objective text.
* **B `shared`**: one backbone, heads R (helpfulness BT) and C with the pairwise cost loss
  `L_C = -log sigmoid(C(y_harmful) - C(y_safer))`, `y_safer` from `safer_response_id`. H rows touch
  backbone + R only, S rows backbone + C only. Objective: 0.5 L_H + 0.5 L_C by equal row counts; both
  terms are single BT losses, so checkpoint selection uses the same 0.5 L_H + 0.5 L_C.
* **C `separate_reward` / `separate_cost`**: two independent backbones from the same checkpoint, R on
  the 10k H rows only (BT), C on the 10k S rows only (same pairwise L_C as B). No parameter sharing.
* **Primary = pairwise only.** Every architecture sees exactly the same pairwise information
  (`better_response_id`, `safer_response_id`). PKU's pointwise `is_response_X_safe` labels are NOT used
  in any primary loss; they appear only as audit descriptives (mean cost of safe vs unsafe responses and
  their shift-invariant gap). The Safe-RLHF-style anchored cost (adds
  `-log sigmoid(s(y0) C(y0)) - log sigmoid(s(y1) C(y1))`, pinning the boundary at C = 0) is kept as an
  ablation behind `cost_pointwise_classification: true` (false in all primary configs).
* **Consequence for constrained GRPO (later stage):** pairwise BT identifies C only up to an additive
  constant, so the constraint threshold must not be hard-coded to 0. Plan: calibrate
  `d = mean cost of the initial Qwen policy on a fixed calibration prompt set` with the same frozen cost
  model, so the dual violation `E[C] - d` is invariant to the shift (group-centered GRPO advantages
  already are).
* `scripts/rm_init_diagnostic.py` reports initialization-only score statistics, initial BT losses and
  the first-batch gradient norm for every architecture (`artifacts/rm_init_diagnostic.{json,md}`).
* Optimization (identical for all, `configs/rm_common.yaml`): AdamW lr 2e-5 (0.9, 0.95, eps 1e-8),
  weight decay 0.1 on matrices only, cosine with 3% warmup, 2 epochs, global batch 64 comparisons =
  per_device 4 x world_size 1 x accumulation 16, grad clip 1.0, bf16 autocast with fp32 master weights,
  gradient checkpointing, lengths 512 / 1024 from the DPO audit, seeds 42/43/44, data_seed 42.
  A and B see 20k rows per epoch (313 steps), each model of C sees 10k (157 steps).
* Validation every 100 steps and at epoch ends; checkpoints best / last / epoch-end (one directory per
  step, `best` is a symlink). Selection: A and B by 0.5 helpfulness loss + 0.5 safety/cost loss, C by
  the model's own validation loss. Metrics per objective x pair type, pair- and prompt-averaged:
  loss, rank loss, strict and tie-adjusted ranking accuracy, margin-sign fractions, mean margin, and
  for cost heads the audit-only mean cost of safe vs unsafe responses and their gap (plus the
  classification loss/accuracy at C = 0 only in the pointwise ablation). `train_summary.json` records parameter counts, tokens per epoch, approximate FLOPs
  (6 x params x tokens) and wallclock.
* Inference: `ScoreModel.load(path, device)` then `score_reward / score_cost / score_scalar(tokenizer,
  prompts, responses, batch_size=...)` (batched, same formatting as training).

```
bash scripts/run_rm_smoke.sh                               # 0.5B smoke of all four trainings
bash scripts/run_rm.sh scalar 42                           # one architecture, one seed (train + eval)
sbatch --export=ARCH=shared,SEED=43 scripts/slurm/rm_seed.sbatch
bash scripts/aggregate_rm.sh scalar                        # -> runs/aggregate_rm_scalar/summary.md
```

## (Constrained) GRPO policy training (Arms 2–4)

Policy and frozen reference pi_ref are both the original Qwen2.5-1.5B-Instruct checkpoint. All three
RL arms share `artifacts/grpo/train_prompts.jsonl` (the 10k training pairs deduplicated by normalized
prompt: 9,054 prompts, one prompt epoch, seed-dependent order), the monitoring prompts, the
generation settings and the policy optimizer (`configs/grpo_common.yaml`).

* Rollout batch: B = 16 prompts x G = 8 completions sampled by the current weights (pi_old) with
  temperature 1, top-p 1, max 512 prompt / 512 completion tokens; completions hitting the cap without
  EOS are flagged truncated (kept, reported). 566 rollout batches per epoch, the last with 14 prompts.
* Advantages per prompt group only (population std + 1e-6, zero-variance -> 0, counted). Scalar arm:
  A = normalized F. Constrained arms (**revision 2026-11-01**): A = A_R - lambda_k * A_C with A_R and A_C each
  normalized separately within the group (the first seed-42 attempt normalized Z = R - lambda C jointly; those
  runs are archived). lambda_k is fixed for the whole rollout batch; the dual update is unchanged (raw cost EMA vs d).
* Gate diagnostic (100 rollout batches, `stop_after_batches`): reports tail mass outside top-20 under full-distribution
  sampling, KL, truncation, length, coherence and a fixed response sample; the automatic go/no-go is hard failures
  (NaN/Inf/OOM) only. Continuations auto-stop only on non-finite values.
* Objective per generated token: min(rho A, clip(rho, 0.8, 1.2) A) - 0.02 KL, rho = pi_theta / pi_old,
  KL = exp(d) - d - 1 with d = log pi_ref - log pi_theta; per-trajectory token mean, then mean over the
  trajectories of the minibatch; prompt and padding masked. Minibatches of 8 whole prompt groups (64
  trajectories) -> 2 optimizer steps per full batch, each trajectory used once, no inner epochs.
* Dual (constrained arms): one global lambda, lambda_0 = 1 (mu = log lambda), after the policy
  updates of each rollout batch: C_ema = C_batch (first) else 0.9 C_ema + 0.1 C_batch, then
  mu += 0.01 lambda_k (C_ema - d). No gradients flow into lambda, R or C.
* Threshold calibration (`scripts/run_grpo_calibrate.sh`): the frozen pi_0 generates 4 responses for
  each of the 256 calibration prompts with fixed seeds (1,024 trajectories, saved once); each constrained
  arm scores the same responses with its own frozen cost model and sets d_arm = mean cost. The pairwise
  cost model is identified only up to a constant, so d absorbs the shift and C = 0 is never treated as a
  safety boundary (no "fraction C > 0" metric). d is fixed across policy seeds; the trainer refuses a
  calibration computed with a different cost checkpoint.
* Monitoring (diagnostic): 256 monitoring prompts x 4 responses with fixed seeds every 100 rollout
  batches and at the end: scalar reward / helpfulness reward, raw cost, d, cost - d, lambda, KL,
  completion length, truncation. The scientific checkpoint is the FINAL policy after one prompt epoch.
* Resume: `runs/<run>/resume/latest.pt` (fp32 policy, optimizer, scheduler, position, RNG, dual state,
  monitoring history) every 25 rollout batches, latest only; rollout sampling seeds are a function of
  (seed, batch index) so resuming neither replays nor skips prompts. Re-running the same command resumes.
* Budget logging per run: prompts, trajectories, generated tokens, optimizer steps, wall-clock,
  GPU-hours, device memory (`train_summary.json`, per-batch `rollout_batches.jsonl`).

```
bash scripts/run_grpo_prepare.sh                     # manifests (idempotent)
bash scripts/run_grpo_calibrate.sh shared            # needs runs/rm_shared_qwen2.5-1.5b_seed42/checkpoints/best
bash scripts/run_grpo_calibrate.sh separate          # needs the separate reward + cost checkpoints (seed 42)
bash scripts/run_grpo.sh scalar 42                   # one seed (resumable); shared / separate likewise
sbatch --export=ARM=shared,SEED=43 scripts/slurm/grpo_seed.sbatch
```

## Common external evaluation (final comparison)

Every method is evaluated at its pre-specified FINAL checkpoint (symmetric rule): DPO after exactly 2 epochs
(`checkpoints/final`), each GRPO arm after exactly 1 prompt epoch (`checkpoints/final`), Base Qwen as is. DPO's
`checkpoints/best` (lowest combined validation loss) is a DIAGNOSTIC artifact only; `eval_generate` verifies each
checkpoint's `trainer_state.json` against its `checkpoint_rule` and refuses any `best` path. All go through one path
with the FROZEN external PKU evaluators `PKU-Alignment/beaver-7b-unified-reward`
(Avg Helpfulness, higher better) and `PKU-Alignment/beaver-7b-unified-cost` (Avg Harmfulness, lower better),
pinned by revision in `configs/eval.yaml`. No training RM/CM is used.

* Prompts: `artifacts/eval/eval_prompts.jsonl` = `test_representative` deduplicated by normalized prompt
  (1,960 prompts, each weighs 1), identical for every method and seed.
* Generation: sampling T = 1.0, top-p 1.0, 512 new tokens, 512 prompt tokens. Each prompt has a fixed seed
  `sha256(eval_seed:prompt_hash)` shared by all methods and policy seeds; sampling uses one CPU generator per
  prompt with inverse-CDF draws, so a prompt's sample does not depend on its batch mates. Base Qwen therefore
  has one response set (its CI comes from prompt resampling only). Responses are saved before scoring.
* Scoring: `BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}</s>`, LLaMA BOS added, score at the
  last token, bf16, batches of 16, both evaluators loaded sequentially per shard (2 x ~13.5 GB).
* Sharding: fixed generation batches of 20 consecutive manifest prompts; a shard is a contiguous range of whole
  batches (`--num-shards N --shard-id i`), so results are identical for any N (N = 4: 500/500/480/480 prompts).
  Scoring reuses the same batches. The merge stage refuses missing/duplicate prompts or mixed shard layouts.
* Statistics: per seed mean over the 1,960 prompts; point estimate = equal-weight mean of seed means; 2,000
  bootstrap replicates resampling prompts WITHIN each seed, averaging the seed means (seeds are never pooled);
  95% CI = 2.5th / 97.5th percentiles; fixed bootstrap seed. Outputs: `runs/eval/summary.{md,json,csv}` with
  seed-level values, aggregates, CIs and checkpoint / evaluator metadata.

```
bash scripts/run_eval_prepare.sh
bash scripts/run_eval_shard.sh base 0 1 0                     # one-shard pilot (generate + score)
for I in 0 1 2 3; do sbatch --export=METHOD=dpo,SEED=42,N=4,I=$I scripts/slurm/eval_shard.sbatch; done
bash scripts/run_eval_merge.sh                                 # verify shards, merge, bootstrap, table
```

## Known limitations / open points

* Single-process trainers: `WORLD_SIZE != 1` is refused. Multi-GPU support (DDP/FSDP) is not implemented; the GRPO
  trainer can place the frozen reference and scorers on other GPUs via `ref_device` / `scorer_device`.
* Apple MPS notes (laptop smoke only): SDPA attention with left-padded generation returns NaN sampling
  probabilities, so `grpo_smoke.yaml` sets `attn_implementation: eager`; a 0.5B policy with fp32 AdamW states
  plus reference and scorer exceeds 16 GB and swaps heavily, so the 0.5B GRPO smoke is impractically slow there.
  The GPU smoke (`run_grpo_smoke.sh` with 1.5B overrides, or the tiny CPU test in `tests/test_grpo.py`) is the
  reference engineering check.
* The Qwen2.5 chat template injects its default system line ("You are Qwen ... a helpful assistant.") for
  every example of both objectives; nothing objective-specific is ever added.
* The data pipeline does not cap pairs per prompt (`max_pairs_per_prompt: null`); the audit reports the
  maximum per split.
