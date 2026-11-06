# Collaborator guide: investigating the shared-backbone reward/cost model

Welcome. This guide points at the code and artifacts you need and sketches investigations we have not done. It is deliberately
open-ended; `docs/EXPERIMENTS.md` has the results and the current diagnosis, `docs/ARTIFACTS.md` says where files live.

## Ground rules

* Scientific settings live in `configs/`; never change them silently. New experiments get new configs and new run names.
* The six-arm study is running on Kempner (`docs/sixarm_spec.md`); do not edit its code or sync to the cluster while it runs.
* Reward-model work needs no GPU for analysis; scoring new text with the 1.5B score models needs one GPU, Beaver scoring two 7B
  evaluators loaded one at a time (fits an 80 GB card; see `scripts/diagnostics/score_eval_responses_with_rms.py`).
* Tests: `python -m pytest tests -q`. Add a test when you add a mechanism.

## Code map for this question

| Topic | Files |
|---|---|
| Reward/cost model architecture (backbone + linear heads, head routing) | `src/saferl/rm_models.py`, `rm_data.py` (ARCHITECTURES table), `train_rm.py` (`microbatch_loss`: one head per row) |
| Losses | `src/saferl/rm_loss.py` (Bradley–Terry; pairwise cost; pointwise anchoring as an off-by-default ablation) |
| RM evaluation on val/test pairs | `src/saferl/eval_rm.py`, `rm_metrics.py` (per objective × pair type); outputs `runs/rm_*/eval/*_rows.jsonl` |
| Frozen scorers as used by RL | `src/saferl/grpo_scoring.py` (`FrozenScorers`: text re-tokenized, EOS appended, raw outputs) |
| Policy training that exploited the head | `src/saferl/train_grpo.py`, `grpo_core.py` |
| Diagnostics used so far | `scripts/diagnostics/*.py` (README table) |
| Style features used in the diagnosis | `style_features` in `src/saferl/train_rebel.py` (therapy-term density, refusal openings, run-on sentences) |

## Proposed investigations (pick, change, or ignore)

**A. Shared-backbone interference.**
* Train the shared model with the cost rows removed (reward head only on the shared code path) and with the safety rows replaced by
  shuffled labels; measure the within-pair length slope and the therapy-term preference of the resulting reward head
  (`rm_length_bias_within_pairs.py`, the style correlation in `shared_rm_pairwise_agreement.py`). If the preference disappears
  without the cost objective, the backbone sharing is the mechanism.
* Probe the backbone: is the final-token representation's "support-resource" direction (fit a linear probe on the therapy-term
  density of base responses) more aligned with the reward head's weight vector in the shared model than in the separate one?
* Vary the mixing ratio of safety rows and the head initialization; small models and short schedules are enough for a trend.

**B. Length bias and reward exploitation.**
* Length-controlled reward-model metrics: accuracy restricted to pairs with |ΔL| ≤ 10 tokens, and the slope of reward on length for
  each architecture, on val/test (data in `runs/rm_*/eval/*_rows.jsonl`; lengths via the tokenizer).
* Counterfactual text: append a fixed therapy-resource paragraph, or pad with neutral text, to base responses and rescore with all
  three reward models (scoring only; `scripts/diagnostics/score_eval_responses_with_rms.py` shows the loading pattern). This
  isolates the style preference from everything else about the shared policy's outputs.
* Best-of-n on base samples under each reward model, judged by Beaver: a cheap, training-free measure of exploitability per model.

**C. Cost-model behaviour.**
* Both learned cost models rated the shared policy's outputs safer than base while Beaver rated them more harmful. Characterize
  what text drives the cost model's score down (ablate refusal sentences, resource lists, length) and whether a cost model trained
  with the pointwise anchoring (`cost_pointwise_classification: true`, an existing ablation) is more robust.
* Calibration: the learned cost is identified only up to a constant; compare learned-cost ranks with Beaver ranks by length bin.

**D. Robustness of the conclusions.**
* Everything is seed 42. Reward-model seeds 43/44 can be trained with the existing scripts (`bash scripts/run_rm.sh shared 43`) to
  see whether the shared head's preference is seed-stable.
* The GRPO runs for seeds 43/44 are prepared (`scripts/slurm/submit_pipeline.sh rest`) but held; they would say whether the
  exploitation is reproducible.

## Practicalities

* Start from `runs/diagnose_shared/scored_eval_*.jsonl`: every evaluation response with all scorer values; join on `prompt_hash` to
  `runs/eval/<method>/seed42/responses.jsonl` for the text.
* The 1,024 calibration responses with scores are `runs/diagnose_shared/scored_pi0_calibration.jsonl`.
* Reward-model checkpoints are small enough to copy (1.5B parameters, ~3 GB bf16 each); policy checkpoints are 3 GB each.
* Keep new outputs under `runs/<your_name>_*` so they are easy to find and never collide with the active runs.
