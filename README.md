# Safe RL on PKU-SafeRLHF

How should separately labelled *helpfulness* and *harmlessness* preferences be represented and used to train a language-model
policy? On Qwen2.5-1.5B-Instruct and the official PKU-SafeRLHF data, the code compares:

| Arm | Reward/cost representation | Policy optimization |
|---|---|---|
| DPO | preference rows from both objectives | sigmoid DPO |
| Scalar RM | one Bradley–Terry model on both objectives | GRPO |
| Shared RM | one backbone, a reward head and a cost head | constrained GRPO (Lagrangian) |
| Separate RMs | an independent reward model and cost model | constrained GRPO (Lagrangian) |

All arms are judged by the same frozen external evaluators (PKU Beaver-7B unified reward and cost) on the same 1,960 held-out
prompts. Results and the diagnosis of the shared-model arm: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Open questions and a code
map for new work: [docs/COLLABORATOR_GUIDE.md](docs/COLLABORATOR_GUIDE.md). Where checkpoints and outputs live:
[docs/ARTIFACTS.md](docs/ARTIFACTS.md). A follow-up study comparing REBEL with GRPO under matched budgets is specified in
[docs/sixarm_spec.md](docs/sixarm_spec.md) and is currently running; its code and configs must not be edited (see below).

## Structure

```
configs/        all scientific settings (YAML with `base:` inheritance; `--set key=value` overrides)
src/saferl/     data preparation, DPO, reward/cost models, (constrained) GRPO, REBEL, external evaluation
scripts/        run_*.sh entry points, diagnostics/, slurm/ job scripts
tests/          pytest suite, CPU only
docs/           experiments, collaborator guide, artifacts, specifications
artifacts/      generated data (untracked, regenerable)
runs/           training runs and evaluation outputs (untracked, on the cluster)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests -q
```

Models and data come from the Hugging Face Hub on first use. PKU-SafeRLHF is CC BY-NC 4.0 (research use only; do not redistribute
derived splits). Qwen2.5 is Apache-2.0; the Beaver evaluators follow the PKU-Alignment licence.

## Reproducing

```bash
python scripts/prepare_data.py --config configs/data.yaml                   # artifacts/ (pinned revision, data_seed 42)
bash scripts/run_dpo.sh 42                                                   # DPO
bash scripts/run_rm.sh shared 42                                             # reward/cost models: scalar | shared | separate_reward | separate_cost
bash scripts/run_grpo_prepare.sh && bash scripts/run_grpo_calibrate.sh shared && bash scripts/run_grpo.sh shared 42
bash scripts/run_eval_prepare.sh && bash scripts/run_eval_shard.sh grpo_shared 42 1 0 && bash scripts/run_eval_merge.sh
```

Every run writes `config_resolved.json` and `train_summary.json`; evaluation writes `summary.{md,json,csv}` with bootstrap
intervals. `scripts/slurm/submit_pipeline.sh` is the job graph that produced the reported results.

## Reward-model diagnostics

`scripts/diagnostics/` contains the scripts behind the shared-model diagnosis: cross-scoring saved responses with every reward and
cost model (`shared_rm_diagnosis.py`), pairwise agreement with the evaluators and length-matched comparisons
(`shared_rm_pairwise_agreement.py`), within-pair length bias (`rm_length_bias_within_pairs.py`), scoring evaluation responses with
the scalar model (`score_eval_responses_with_rms.py`), and a tail-mass probe of a policy (`grpo_tail_mass.py`). `saferl.eval_rm`
evaluates any score-model checkpoint on the validation and test pairs.

## Active experiment

The six-arm study runs unattended on the cluster and reads its code at job start. Do not modify `configs/sixarm/`,
`configs/rebel_common.yaml`, `configs/grpo_common.yaml`, `configs/eval*.yaml`, `src/saferl/train_rebel.py`, `train_grpo.py`,
`rebel_core.py`, `grpo_core.py`, `grpo_rollout.py`, `grpo_scoring.py`, `eval_*.py`, `scripts/sixarm_*`, `scripts/run_sixarm_*`,
`scripts/eval_pairs.py` or `scripts/slurm/` until it finishes, and do not sync to the cluster.
