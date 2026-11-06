# Experiments so far: the four-arm study (seed 42) and the shared-model diagnosis

All numbers are from policy seed 42 with reward-model seed 42. Confidence intervals are 2,000-replicate bootstraps over the 1,960
evaluation prompts; they do not include training-seed variation (seeds 43/44 were prepared but held back).

## 1. Setup (frozen)

* **Data.** Official PKU-SafeRLHF, pinned revision `9421ffaf…`; prompts shared with any official test prompt removed from the training
  pool; 10,000 training pairs (5,000 agreement + 5,000 conflict), 2,000 val, 2,000 test_balanced, 2,000 test_representative;
  data_seed 42. Agreement pairs: the same response wins both objectives; conflict: the winners differ.
* **Policy.** Qwen2.5-1.5B-Instruct, full fine-tuning, fp32 master weights + bf16 autocast, official chat template.
* **DPO.** Sigmoid, β 0.1, lr 5e-7, batch 64, 2 epochs, both objectives as separate rows without objective labels. Primary checkpoint = final.
* **Reward/cost models.** Bradley–Terry, linear head on the final token, lr 2e-5, batch 64, 2 epochs, best-by-val checkpoint.
  Cost loss is pairwise only (C identified up to a constant). The shared model mixes helpfulness and safety rows 50/50 in one
  shuffle; each row updates the backbone and only its own head.
* **GRPO.** 16 prompts × 8 samples per batch, 566 batches = one prompt epoch over 9,054 unique training prompts, 2 optimizer steps per
  batch; clipped surrogate ε 0.2, per-token k3 KL to π0 with β 0.02, lr 1e-6 cosine; sampling T 1, top_p 1, **top_k 0 pinned**
  (Qwen's packaged `top_k=20` leaked into a first run and was fixed), 512 new tokens.
* **Constrained GRPO.** A = A_R − λA_C with R and C group-normalized separately; log-λ dual (λ0 1, step 0.01, EMA 0.9) once per batch;
  d = mean initial-policy cost on 256 calibration prompts × 4 samples under the arm's own cost model:
  d_shared = −1.726 (σ 2.08), d_separate = −4.172 (σ 1.80).
* **Evaluation.** 1,960 deduplicated test_representative prompts, one sample each with per-prompt seeds shared by every method,
  Beaver-7B unified reward (helpfulness ↑) and cost (harmfulness ↓).

Reward-model validation accuracy (val, tie-adjusted): scalar 61.3 % helpfulness / 66.5 % safety; shared 66.7 % / 71.2 %;
separate reward 63.7 %, separate cost 72.0 %.

## 2. Main result

| Method | Helpfulness ↑ [95 % CI] | Harmfulness ↓ [95 % CI] | Truncated at 512 | Mean length |
|---|---|---|---|---|
| Base Qwen | −2.33 [−2.57, −2.07] | −5.11 [−5.23, −4.99] | 3.5 % | 135 |
| DPO | −2.09 [−2.34, −1.83] | −5.26 [−5.38, −5.15] | 2.7 % | — |
| Scalar RM + GRPO | **4.13** [4.07, 4.19] | −8.78 [−8.84, −8.73] | 85.6 % | 493 |
| Shared constrained GRPO | 2.82 [2.77, 2.87] | **−4.05** [−4.11, −3.99] (worse than base) | 97.5 % | 505 |
| Separate constrained GRPO | 3.59 [3.53, 3.65] | **−8.86** [−8.92, −8.81] | 2.4 % | 328 |

All GRPO policies are coherent under full-distribution sampling (tail mass outside top-20 stable at 5–9 %). End of training:
scalar KL 0.30; shared λ 0.074, cost −3.22 vs d −1.73, KL 0.32; separate λ 0.049, cost −7.91 vs d −4.17, KL 0.57. In both constrained
arms the cost fell below d within ~15–35 batches and λ decayed to < 0.1, so the constraint was slack for most of training.

Where the runs live: `docs/ARTIFACTS.md`. Earlier runs with the top-k leak and with jointly normalized advantages are archived
(`runs/archive_topk20_*`, `runs/archive_jointnorm_*`) and are not valid evidence.

## 3. Diagnosis of the shared arm (existing artifacts only)

Scripts: `scripts/diagnostics/` (see README). Outputs: `runs/diagnose_shared/` on the cluster (report.json, part2_report.json,
scored_eval_*.jsonl with every scorer on every evaluation response, part1_examples.txt, plots, length_bias_train_agreement.*).

1. **Not an implementation bug.** Every logged λ update recomputes to ~1e-16; rescoring the calibration set reproduces d exactly.
2. **Not a held-out ranking deficit.** On identical val pairs the shared head is +3.0 points better than the separate model on
   helpfulness (CI excludes 0) and tied on harmlessness (71.3 % vs 72.0 %); the two cost scorers rank responses almost identically
   (Spearman 0.91). Before RL, the shared head agrees with Beaver *more* than the separate model does (0.77–0.81 vs 0.59–0.61 on
   initial-policy responses; 78 % vs 75 % sign agreement on val pairs).
3. **The policy exploited the reward head.** Pooled percentile ranks on the 7,840 evaluation responses: the shared head ranks the
   shared policy's outputs at the 87th percentile, Beaver at the 45th (gap +0.42); the separate reward model's self-overvaluation is
   +0.29, the scalar model's +0.12. The separate and scalar models place the shared outputs where Beaver does (gaps +0.06 / +0.01).
4. **Length is not the explanation by itself.** Among 1,637 prompts where both the shared and scalar policies hit the 512-token
   cap, Beaver harmfulness is 4.6 points worse for the shared policy on 99.8 % of prompts, and every learned cost model agrees.
   After controlling for length, the shared head's agreement with Beaver on base outputs drops from 0.81 to 0.42 (separate 0.60).
5. **What the shared head rewards.** The shared policy writes a one-line refusal followed by a run-on list of therapy / counseling /
   mindfulness resources regardless of the prompt (8.4 therapy terms per 100 words vs 2.2 separate, 1.5 scalar, 0.1 base; longest
   sentence 212 words). Pooled correlation of each helpfulness scorer with that term density after length control: shared head
   **+0.70**, separate +0.28, scalar −0.32, Beaver −0.27. The preference is latent in the frozen head (invisible on base outputs,
   where such text is rare) and revealed by optimization.
6. **Preference-data length bias.** Helpfulness labels prefer the longer response in 63 % of training pairs (median +14 tokens);
   safety prefers the shorter one on conflict pairs only. Within agreement pairs, reward gain per +100 tokens: shared head 1.09,
   separate 0.68, at equal pair accuracy (76 %).
7. **Both learned cost models were fooled** by the shared policy's outputs (rated safer than base while Beaver rates them more
   harmful), roughly equally; their agreement with Beaver fell from 0.65 to ~0.4 Spearman.

## 4. Open questions

* Why does the shared backbone's reward head carry the support-resource / therapy preference and the length preference more strongly
  than an independently trained reward model on the same helpfulness labels? Candidate mechanisms: interference from the cost
  objective through the shared backbone (safety-preferred responses often redirect to help resources), capacity sharing, or label
  noise interactions. The scalar model, trained on the same safety comparisons through a single head, shows the opposite sign.
* Is the exploitability intrinsic to the head, or a property of any reward model that is the optimization target? Cross-scoring
  shows the separate reward model does not overvalue the shared outputs, which argues for the former.
* The constraint went slack in both constrained arms; would a binding constraint (stricter d) or a cost model robust to the style
  change the picture? (The six-arm study varies d but keeps the separate cost model.)
* No truncation penalty exists; two arms wrote to the cap. Length-controlled evaluation would help interpret helpfulness.
* Everything is one training seed.
