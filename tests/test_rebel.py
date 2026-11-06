"""REBEL: regression correctness, log-probabilities, constraints, training integrity (tiny CPU end-to-end), configs."""
import copy
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from saferl.rebel_core import AdditiveDual, exponential_tilt, kl_stats, pair_indices, raw_threshold, rebel_loss, standardize
from saferl.utils import load_config, read_json

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- regression correctness

def test_loss_at_theta_t_is_mean_target_squared():
    torch.manual_seed(0)
    lp = torch.randn(10); z = torch.randn(5)
    out = rebel_loss(lp[:5], lp[5:], lp[:5], lp[5:], z, eta=0.7)       # theta = theta_t: relative change exactly zero
    assert math.isclose(float(out["loss"]), float((z ** 2).mean()), rel_tol=1e-6)
    assert out["pred_std"] == 0.0 and out["n_pairs"] == 5


def test_swapping_pair_members_leaves_loss_unchanged():
    torch.manual_seed(1)
    n1, n2, o1, o2, z = (torch.randn(6) for _ in range(5))
    a = rebel_loss(n1, n2, o1, o2, z, 1.3)["loss"]; b = rebel_loss(n2, n1, o2, o1, -z, 1.3)["loss"]
    assert torch.allclose(a, b)


def test_mean_of_squared_residuals_not_square_of_mean_and_one_over_eta_scaling():
    o = torch.zeros(2); n = torch.tensor([1.0, -1.0], requires_grad=True); z = torch.zeros(2)
    out = rebel_loss(n, torch.zeros(2), o, o, z, eta=1.0)
    assert math.isclose(float(out["loss"]), 1.0)                         # residuals +1, -1 -> mean of squares 1 (square of mean would be 0)
    # pred = dl / eta: with dl = 2 and z = 1, eta = 2 fits exactly, eta = 1 does not; gradient = 2 resid / (eta n)
    n2 = torch.tensor([2.0, 2.0], requires_grad=True)
    assert math.isclose(float(rebel_loss(n2, torch.zeros(2), o, o, torch.ones(2), eta=2.0)["loss"]), 0.0)
    l1 = rebel_loss(n2, torch.zeros(2), o, o, torch.ones(2), eta=1.0)
    l1["loss"].backward()
    assert torch.allclose(n2.grad, 2 * l1["residual"] / (1.0 * 2))


def test_target_and_old_logprobs_are_detached_gradients_reach_only_the_policy():
    n1 = torch.randn(4, requires_grad=True); n2 = torch.randn(4, requires_grad=True)
    o1 = torch.randn(4, requires_grad=True); o2 = torch.randn(4, requires_grad=True); z = torch.randn(4, requires_grad=True)
    rebel_loss(n1, n2, o1, o2, z, 1.0)["loss"].backward()
    assert n1.grad is not None and n2.grad is not None
    assert o1.grad is None and o2.grad is None and z.grad is None


def test_tabular_exact_regression_recovers_exponential_tilt():
    """Full pair coverage on a finite action set: minimizing Eq. 1 exactly gives pi_{t+1} proportional to pi_t exp(eta r)."""
    torch.manual_seed(0)
    K, eta = 6, 0.8
    logits_t = torch.randn(K); pi_t = torch.softmax(logits_t, 0); old = torch.log(pi_t); r = torch.randn(K)
    i, j = zip(*[(a, b) for a in range(K) for b in range(K) if a < b])
    i, j = torch.tensor(i), torch.tensor(j)
    theta = logits_t.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=0.5, max_iter=200, tolerance_grad=1e-10, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        lp = torch.log_softmax(theta, 0)
        loss = rebel_loss(lp[i], lp[j], old[i], old[j], r[i] - r[j], eta)["loss"]
        loss.backward()
        return loss
    for _ in range(5):
        opt.step(closure)
    assert torch.allclose(torch.softmax(theta.detach(), 0), exponential_tilt(pi_t, r, eta), atol=1e-4)
    assert float(closure()) < 1e-8


def test_pairs_fixed_disjoint_cover_all_and_ignore_scores():
    pairs = pair_indices(3, 8, seed=42, batch_index=5)
    assert len(pairs) == 12 and pairs == pair_indices(3, 8, 42, 5)
    for g in range(3):
        members = sorted(x for pr in pairs[g * 4:(g + 1) * 4] for x in pr)
        assert members == list(range(g * 8, g * 8 + 8))
    assert pairs != pair_indices(3, 8, 42, 6)
    assert "score" not in inspect.signature(pair_indices).parameters and "reward" not in inspect.signature(pair_indices).parameters
    with pytest.raises(ValueError):
        pair_indices(1, 7, 0, 0)


# ----------------------------------------------------------------------------- constraints / dual

def test_dual_direction_projection_and_logging():
    d = AdditiveDual(d=-0.5, beta=0.01, lam=1.0)
    r = d.update(0.0)                                                     # cost above d -> violation +0.5 -> lambda increases
    assert r["violation"] == 0.5 and math.isclose(d.lam, 1.005) and not r["projected"]
    r = d.update(-2.0)                                                    # cost below d -> lambda decreases
    assert r["violation"] == -1.5 and math.isclose(d.lam, 1.005 - 0.015)
    d2 = AdditiveDual(d=-0.5, beta=0.01, lam=0.004)
    r = d2.update(-2.0)
    assert d2.lam == 0.0 and r["projected"] and r["lambda_unprojected"] < 0 and r["log_lambda"] is None
    assert d2.update(-2.0)["lambda"] == 0.0                               # stays non-negative
    with pytest.raises(FloatingPointError):
        d2.update(float("nan"))
    sd = d.state_dict(); assert AdditiveDual.from_state_dict(sd).lam == d.lam
    with pytest.raises(ValueError):
        AdditiveDual(d=0.0, beta=0.1, lam=-1.0)


def test_standardized_and_raw_thresholds_are_the_same_constraint():
    torch.manual_seed(2)
    mu, sigma, d_std = 3.2, 1.7, -0.5
    c = torch.randn(1000) * sigma + mu
    cs = standardize(c, mu, sigma)
    assert torch.equal(cs <= d_std, c <= raw_threshold(mu, sigma, d_std) + 1e-6 * sigma) or int((cs <= d_std).sum()) == int((c <= raw_threshold(mu, sigma, d_std)).sum())
    assert math.isclose(float(cs.mean()), (float(c.mean()) - mu) / sigma, abs_tol=1e-5)
    assert math.isclose(raw_threshold(mu, sigma, -0.5), mu - 0.5 * sigma) and math.isclose(raw_threshold(mu, sigma, 0.0), mu) and math.isclose(raw_threshold(mu, sigma, -1.0), mu - sigma)
    with pytest.raises(ValueError):
        standardize(c, mu, 0.0)


def test_kl_estimators_zero_for_identical_and_nonnegative():
    torch.manual_seed(3)
    p = [torch.randn(5) - 3, torch.randn(7) - 3]
    k = kl_stats(p, p)
    assert k["kl_k3_token_mean"] == 0.0 and k["kl_k3_seq_mean"] == 0.0 and k["n_tokens"] == 12 and k["finite"]
    q = [x - 0.3 * torch.rand_like(x) for x in p]
    assert kl_stats(p, q)["kl_k3_token_mean"] > 0


# ----------------------------------------------------------------------------- log-probabilities (tiny real-vocab model)

def _traj(tokenizer, tiny_model, prompt, n_new, seed):
    from saferl.grpo_rollout import generate_completions
    return generate_completions(tiny_model, tokenizer, [prompt], 1, 64, n_new, "cpu", "fp32", seed=seed)[0]


@pytest.mark.hf
def test_sequence_logprob_sum_causal_shift_mask_eos_and_padding(tokenizer, tiny_model):
    from saferl.grpo_rollout import batched_token_logprobs, collate_trajectories, token_logprobs
    t1 = _traj(tokenizer, tiny_model, "Hello there", 4, 0); t2 = _traj(tokenizer, tiny_model, "A much longer prompt about safety and help.", 9, 1)
    t2["completion_ids"] = t2["completion_ids"] + ([tokenizer.eos_token_id] if t2["completion_ids"][-1] != tokenizer.eos_token_id else [])
    assert len(t1["completion_ids"]) != len(t2["completion_ids"])
    b = collate_trajectories([t1, t2], tokenizer.pad_token_id)
    r = token_logprobs(tiny_model, b, "cpu", "fp32")
    for k, t in enumerate((t1, t2)):
        ids = torch.tensor([t["prompt_ids"] + t["completion_ids"]])
        with torch.no_grad():
            lp = torch.log_softmax(tiny_model(input_ids=ids).logits[0, :-1].float(), -1)   # logits[t] predicts ids[t+1]
        P = len(t["prompt_ids"])
        manual = sum(float(lp[P - 1 + j, tok_id]) for j, tok_id in enumerate(t["completion_ids"]))   # every completion token incl. EOS, no prompt token
        assert math.isclose(float(r["sum_logprob"][k]), manual, rel_tol=1e-4, abs_tol=1e-4)
        assert int(r["num_tokens"][k]) == len(t["completion_ids"])
        assert torch.all(r["token_logprobs"][k] <= 1e-5)                                   # valid log-probabilities
    # padded batch == unpadded single; batched helper == teacher-forced sums
    single = token_logprobs(tiny_model, collate_trajectories([t2], tokenizer.pad_token_id), "cpu", "fp32")["sum_logprob"][0]
    assert math.isclose(float(single), float(r["sum_logprob"][1]), rel_tol=1e-4, abs_tol=1e-4)
    stored = batched_token_logprobs(tiny_model, [t1, t2], tokenizer.pad_token_id, 1, "cpu", "fp32")
    for k in range(2):
        assert math.isclose(float(stored[k].sum()), float(r["sum_logprob"][k]), rel_tol=1e-4, abs_tol=1e-4)


def test_trainer_uses_sequence_sums_never_per_token_means_and_never_raw_generate():
    src = inspect.getsource(__import__("saferl.train_rebel", fromlist=["REBELTrainer"]))
    assert 'r["sum_logprob"]' in src and 'num_tokens' not in src.split("def primal_steps")[1].split("def rollout_batch")[0]
    assert "model.generate(" not in src and "generate_completions(" in src                    # top_k=0 pinned inside generate_completions
    assert "batched_token_logprobs(self.policy, trajs" in src.split("def rollout_batch")[1].split("agg = self.primal_steps")[0]   # old log-probs stored BEFORE the update
    from saferl.dpo_loss import sequence_logprobs
    # per-token log-ratio c on n tokens -> sequence difference n*c (sum), i.e. longer sequences are not down-weighted
    logits = torch.zeros(1, 5, 3); logits[0, :, 0] = 1.0
    ids = torch.zeros(1, 5, dtype=torch.long); mask = torch.tensor([[0, 1, 1, 1, 1]])
    out = sequence_logprobs(logits, ids, mask)
    assert math.isclose(float(out["sum_logprob"]), 4 * float(torch.log_softmax(logits[0, 0], -1)[0]), rel_tol=1e-6)


# ----------------------------------------------------------------------------- data: leakage-free manifests

def test_rebel_manifests_are_prompt_disjoint_and_stratified(smoke_artifacts_dir, tmp_path):
    from conftest import synthetic_pool
    from saferl.rebel_data import build_rebel_manifests, load_rebel_prompts, prompt_key, used_prompt_keys
    import shutil

    art = tmp_path / "art"; shutil.copytree(smoke_artifacts_dir, art)
    train = synthetic_pool(400, "train", seed=1, n_prompts=200)
    test = synthetic_pool(40, "test", seed=2, n_prompts=20)
    forbidden_text = json.loads((art / "splits" / "train.pairs.jsonl").read_text().splitlines()[0])["prompt"]
    train.append({**train[0], "prompt": forbidden_text, "pair_id": "zzzz_forbidden"})      # a split prompt sneaked into the pool
    ref = {"Alpaca3-70B": 60, "Beavertails": 25, "WizardLM-30B-Uncensored": 15}
    info = build_rebel_manifests(str(art), train, test, data_seed=42, n_train=40, n_calibration=8, n_monitoring=8, reference_sources=ref)
    assert info["counts"] == {"train": 40, "calibration": 8, "monitoring": 8}
    assert all(v == 0 for v in info["hash_intersections"].values())
    assert info["by_source"]["train"] == {"Alpaca3-70B": 24, "Beavertails": 10, "WizardLM-30B-Uncensored": 6}
    keys = {n: {p["prompt_hash"] for p in load_rebel_prompts(str(art), n)} for n in ("train", "calibration", "monitoring")}
    used = set().union(*used_prompt_keys(str(art)).values()) | {prompt_key(p["prompt"]) for p in test}
    for n, k in keys.items():
        assert not (k & used)
    assert prompt_key(forbidden_text) not in keys["train"] | keys["calibration"] | keys["monitoring"]
    assert not (keys["train"] & keys["calibration"]) and not (keys["train"] & keys["monitoring"]) and not (keys["calibration"] & keys["monitoring"])
    info2 = build_rebel_manifests(str(art), train, test, data_seed=42, n_train=40, n_calibration=8, n_monitoring=8, reference_sources=ref)
    assert info2["hashes"] == info["hashes"]                                                 # deterministic


# ----------------------------------------------------------------------------- configs

def test_arm_configs_share_common_hyperparameters_and_eta_is_not_the_lr():
    from saferl.train_rebel import COMMON_POLICY_KEYS
    cfgs = {a: load_config(str(ROOT / "configs" / f)) for a, f in (("scalar", "rebel_scalar.yaml"), ("shared", "rebel_shared_constrained.yaml"), ("separate", "rebel_separate_constrained.yaml"))}
    for k in COMMON_POLICY_KEYS:
        vals = {json.dumps(c[k], sort_keys=True) for c in cfgs.values()}
        assert len(vals) == 1, k
    c = cfgs["scalar"]
    assert c["rebel_eta"] != c["learning_rate"] and c["kl_to_initial_coef"] == 0 and c["d_std"] == -0.5 and c["lambda_init"] == 1.0 and c["dual_step_size"] == 0.01
    assert c["generation"]["temperature"] == 1.0 and c["generation"]["top_p"] == 1.0 and c["generation"]["max_completion_length"] == 512
    assert c["group_size"] == 8 and c["prompt_batch_size"] == 16 and c["pairs_per_minibatch"] == 32 and c["inner_epochs"] == 1
    for a, cc in cfgs.items():
        assert all("_seed42/" in p for p in cc["scorer_checkpoints"].values()) and cc["scorer_rm_seed"] == 42
    ev = load_config(str(ROOT / "configs" / "eval_rebel.yaml"))
    for m in ("rebel_scalar", "rebel_shared", "rebel_separate"):
        assert ev["methods"][m]["checkpoint_rule"] == "rebel_final_1_prompt_epoch" and ev["methods"][m]["seeds"] == [42]
    assert ev["generation"] == load_config(str(ROOT / "configs" / "eval.yaml"))["generation"]
    m = load_config(str(ROOT / "configs" / "grpo_matched_shared_constrained.yaml"))
    g = load_config(str(ROOT / "configs" / "grpo_shared_constrained.yaml"))
    for k in ("clip_epsilon", "beta_kl", "learning_rate", "group_size", "prompt_batch_size", "groups_per_minibatch", "generation"):
        assert m[k] == g[k]
    assert m["d_std"] == -0.5 and m["dual_step_size"] == 0.01


# ----------------------------------------------------------------------------- tiny end-to-end (CPU)

def _tiny_setup(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path, arm="shared"):
    from conftest import synthetic_pool
    from saferl.rebel_calibrate import calibrate_arm, generate_calibration
    from saferl.rebel_data import build_rebel_manifests
    from saferl.rm_models import ScoreModel
    import shutil

    art = tmp_path / "art"; shutil.copytree(smoke_artifacts_dir, art)
    build_rebel_manifests(str(art), synthetic_pool(200, "train", seed=1, n_prompts=120), synthetic_pool(20, "test", seed=2, n_prompts=10),
                          data_seed=42, n_train=6, n_calibration=3, n_monitoring=4, reference_sources={"Alpaca3-70B": 1})
    pol_dir = tmp_path / "policy"; tiny_model.save_pretrained(pol_dir); tokenizer.save_pretrained(pol_dir)
    torch.manual_seed(1)
    heads = {"shared": ["reward", "cost"], "scalar": ["scalar"]}[arm]
    rm = ScoreModel(copy.deepcopy(tiny_model.model), heads, tiny_model.config.hidden_size, 0.02, str(pol_dir))
    rm_dir = tmp_path / "rm"; rm.save_pretrained(rm_dir, "float32"); tokenizer.save_pretrained(rm_dir)
    base = load_config(str(ROOT / "configs" / "rebel_smoke.yaml"))
    cfg = {**base, "arm": arm, "artifacts_dir": str(art), "policy_model": str(pol_dir), "device": "cpu", "precision": "fp32", "gradient_checkpointing": False,
           "attn_implementation": None, "prompt_batch_size": 2, "group_size": 4, "pairs_per_minibatch": 2, "policy_micro_batch_size": 2,
           "generation": {"max_prompt_length": 128, "max_completion_length": 6, "temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0, "do_sample": True},
           "max_rollout_batches": 3, "eval_every_batches": 2, "monitor_at_start": True, "monitor_responses_per_prompt": 1, "monitor_gen_batch": 8,
           "dump_every_batches": 1, "dump_n_prompts": 2, "checkpoint_every_batches": 2, "resume_every_batches": 2, "learning_rate": 1e-3, "rebel_eta": 1.0,
           "scorer_checkpoints": {arm: str(rm_dir)} if arm == "shared" else {"scalar": str(rm_dir)}, "calibration_file": str(tmp_path / f"cal_{arm}.json"),
           "runs_dir": str(tmp_path / "runs_a"), "run_name": "tiny", "seed": 7, "diagnostic": {"n_samples": 2, "seed": 1}, "collapse_check_every": 1}
    gen = tmp_path / "cal_gen.jsonl"
    if not gen.exists():
        generate_calibration(cfg, gen, seeds=[1, 2], gen_batch=8)
    rep = calibrate_arm(cfg, gen, tmp_path / f"cal_{arm}.json")
    return cfg, rep


@pytest.mark.hf
@pytest.mark.artifacts
def test_tiny_rebel_run_invariants_and_resume(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    from safetensors.torch import load_file
    from saferl.eval_common import verify_checkpoint_rule
    from saferl.train_rebel import REBELTrainer

    cfg, rep = _tiny_setup(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path, "shared")
    assert rep["n_trajectories"] == 6 and rep["heads"]["cost"]["sigma"] > 0 and math.isclose(rep["d_raw"], rep["heads"]["cost"]["mu"] - 0.5 * rep["heads"]["cost"]["sigma"])
    assert rep["reference_thresholds"]["alpha=0.0"]["d_raw"] == rep["heads"]["cost"]["mu"]
    s_a = REBELTrainer(cfg).train()
    rd = Path(s_a["run_dir"])
    assert s_a["n_prompts"] == 6 and s_a["n_trajectories"] == 24 and s_a["n_pairs"] == 12 and s_a["opt_step"] == 6 and s_a["n_dual_updates"] == 3
    rows = [json.loads(l) for l in open(rd / "rollout_batches.jsonl")]
    assert [r["batch_index"] for r in rows] == [0, 1, 2]
    for r in rows:
        assert r["n_pairs"] == 4 and r["optimizer_steps"] == 2 and r["finite"]
        assert math.isclose(r["loss_at_theta_t"], r["target_sq_mean"], rel_tol=1e-5)                      # loss == mean z^2 before any step
        assert math.isclose(r["lambda"], max(0.0, r["lambda_used"] + 0.01 * (r["cost_hat"] - r["d"])), rel_tol=1e-9)   # additive projected dual, standardized units
        assert math.isclose(r["cost_hat"], r["cost_std_mean"], rel_tol=1e-9) and r["d"] == -0.5             # dual uses the batch mean standardized cost
        assert r["old_logprob_sum_mean"] <= 0 and r["kl_initial"]["kl_k3_token_mean"] >= 0 and r["kl_prev"]["kl_k3_token_mean"] >= 0
    assert rows[1]["lambda_used"] == rows[0]["lambda"]                                                     # lambda_t fixed for the whole batch, updated after it
    # artifacts: monitor (+ samples), dumps, intermediate + final checkpoints, resume; eval rule accepts final, refuses intermediate
    assert (rd / "monitor" / "start_samples.jsonl").exists() and (rd / "monitor" / "batch_00002.json").exists() and (rd / "monitor" / "final.json").exists()
    assert (rd / "generations" / "batch_00000.jsonl").exists() and (rd / "generations" / "batch_00001.jsonl").exists()
    assert (rd / "checkpoints" / "batch_00002" / "trainer_state.json").exists() and (rd / "checkpoints" / "final" / "trainer_state.json").exists()
    assert verify_checkpoint_rule(str(rd / "checkpoints" / "final"), "rebel_final_1_prompt_epoch")["trainer_state"]["n_prompts"] == 6
    with pytest.raises(ValueError):
        verify_checkpoint_rule(str(rd / "checkpoints" / "batch_00002"), "rebel_final_1_prompt_epoch")
    assert read_json(rd / "checkpoints" / "final" / "trainer_state.json")["algorithm"] == "rebel"
    # interrupted run: stop after 2 batches (resume state at batch 2), then resume -> identical weights, lambda and counters
    cfg_b = {**cfg, "runs_dir": str(tmp_path / "runs_b"), "max_rollout_batches": 2}
    REBELTrainer(cfg_b).train()
    s_b = REBELTrainer({**cfg_b, "max_rollout_batches": 3}).train()
    assert s_b["resumed"] and s_b["n_prompts"] == 6 and s_b["n_pairs"] == 12 and s_b["opt_step"] == 6 and s_b["n_dual_updates"] == 3
    assert math.isclose(s_a["final_lambda"], s_b["final_lambda"], rel_tol=1e-9)
    wa = load_file(Path(s_a["final_checkpoint"]) / "model.safetensors"); wb = load_file(Path(s_b["final_checkpoint"]) / "model.safetensors")
    for k in wa:
        assert torch.allclose(wa[k].float(), wb[k].float(), atol=1e-5, rtol=1e-4), k
    rows_b = [json.loads(l) for l in open(Path(s_b["run_dir"]) / "rollout_batches.jsonl")]
    assert [r["batch_index"] for r in rows_b] == [0, 1, 2]                                                  # no duplicated / dropped batch


@pytest.mark.hf
@pytest.mark.artifacts
def test_old_logprobs_fixed_across_minibatches_and_scorers_receive_no_gradient(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    from saferl.grpo_rollout import batched_token_logprobs, generate_completions
    from saferl.rebel_core import pair_indices
    from saferl.train_rebel import REBELTrainer

    cfg, _ = _tiny_setup(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path, "shared")
    tr = REBELTrainer({**cfg, "runs_dir": str(tmp_path / "runs_c")}); tr.setup()
    prompts = [p["prompt"] for p in tr.train_prompts[:2]]
    trajs = generate_completions(tr.policy, tr.tok, prompts, 4, 128, 6, "cpu", "fp32", seed=3)
    old = torch.stack([x.sum() for x in batched_token_logprobs(tr.policy, trajs, tr.pad, 2, "cpu", "fp32")])
    old_copy = old.clone()
    pairs = pair_indices(2, 4, 7, 0); target = torch.randn(len(pairs))
    before = [p.detach().clone() for p in tr.policy.parameters()]
    agg = tr.primal_steps(trajs, old, pairs, target)
    assert agg["steps"] == 2 and torch.equal(old, old_copy)                                                 # stored old log-probs untouched by the steps
    assert any(not torch.equal(a, b) for a, b in zip(before, tr.policy.parameters()))                       # policy moved at lr > 0
    new = torch.stack([x.sum() for x in batched_token_logprobs(tr.policy, trajs, tr.pad, 2, "cpu", "fp32")])
    assert not torch.allclose(new, old)                                                                     # recomputing would have changed them
    for m in tr.scorers.models.values():
        assert all((not p.requires_grad) and p.grad is None for p in m.parameters())
    assert all(p.grad is None for p in tr.ref.parameters())
    tr.batch_log.close()


@pytest.mark.hf
@pytest.mark.artifacts
def test_hard_failure_on_invalid_dual_and_gate_stop_resume(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    from saferl.train_rebel import REBELTrainer

    cfg, _ = _tiny_setup(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path, "shared")
    s1 = REBELTrainer({**cfg, "runs_dir": str(tmp_path / "runs_g"), "stop_after_batches": 2}).train()
    rd = Path(s1["run_dir"])
    assert s1["stopped_for_diagnostic"] and s1["gates_passed"] and s1["next_batch_index"] == 2 and not (rd / "checkpoints" / "final").exists()
    d = read_json(rd / "diagnostic" / "batch_00002.json")
    assert d["finite"] and 0 <= d["tail_mass"] <= 1 and "style_therapy_terms_per_100w" in d
    s2 = REBELTrainer({**cfg, "runs_dir": str(tmp_path / "runs_g")}).train()
    assert s2["resumed"] and s2["n_pairs"] == 12 and (rd / "checkpoints" / "final").exists()
    with pytest.raises(ValueError):                                                                         # config d_std must equal the calibration's
        REBELTrainer({**cfg, "runs_dir": str(tmp_path / "runs_h"), "d_std": -1.0}).train()


@pytest.mark.hf
@pytest.mark.artifacts
def test_scalar_arm_has_no_dual_and_matched_grpo_uses_additive_dual_in_standardized_units(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    from saferl.grpo_matched import GRPOMatchedTrainer
    from saferl.train_rebel import REBELTrainer

    cfg, rep = _tiny_setup(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path, "scalar")
    s = REBELTrainer({**cfg, "runs_dir": str(tmp_path / "runs_s"), "max_rollout_batches": 2}).train()
    rows = [json.loads(l) for l in open(Path(s["run_dir"]) / "rollout_batches.jsonl")]
    assert s["final_lambda"] is None and "lambda" not in rows[0] and "scalar_std_mean" in rows[0] and s["n_pairs"] == 8
    cfg_sh, _ = _tiny_setup(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path / "sh", "shared")
    gbase = load_config(str(ROOT / "configs" / "grpo_smoke.yaml"))
    gcfg = {**gbase, **{k: cfg_sh[k] for k in ("arm", "artifacts_dir", "policy_model", "device", "precision", "gradient_checkpointing", "attn_implementation", "generation",
                                              "scorer_checkpoints", "calibration_file", "seed")},
            "prompt_batch_size": 2, "group_size": 4, "groups_per_minibatch": 1, "policy_micro_batch_size": 4, "max_rollout_batches": 2, "eval_every_batches": 100,
            "monitor_at_start": False, "resume_every_batches": 100, "learning_rate": 1e-3, "monitor_responses_per_prompt": 1, "monitor_gen_batch": 8,
            "runs_dir": str(tmp_path / "runs_m"), "run_name": "matched", "d_std": -0.5, "lambda_init": 1.0, "dual_step_size": 0.01, "dual_projection": True}
    sm = GRPOMatchedTrainer(gcfg).train()
    mrows = [json.loads(l) for l in open(Path(sm["run_dir"]) / "rollout_batches.jsonl")]
    for r in mrows:
        assert math.isclose(r["lambda"], max(0.0, r["lambda_used"] + 0.01 * (r["cost_batch_mean"] - (-0.5))), rel_tol=1e-9) and r["d"] == -0.5
    assert read_json(Path(sm["run_dir"]) / "config_resolved.json")["algorithm"] == "grpo_matched"
    assert sm["n_prompts"] == 4 and sm["opt_step"] == 4
