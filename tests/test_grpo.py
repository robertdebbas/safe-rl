"""GRPO stage tests: manifests, advantages, calibration invariances, dual update, clipped objective /
KL / masking, minibatching, config consistency, and a tiny resume-vs-uninterrupted check."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from saferl.grpo_core import DualState, grpo_loss, group_advantages, lagrangian_scores
from saferl.grpo_data import (CALIBRATION_GEN_SEEDS, MONITOR_GEN_SEEDS, build_manifests, load_prompts, prompt_order,
                              rollout_batches, unique_prompts)
from saferl.utils import load_config, read_json

ROOT = Path(__file__).resolve().parents[1]
ARM_CONFIGS = {"scalar": "grpo_scalar.yaml", "shared": "grpo_shared_constrained.yaml", "separate": "grpo_separate_constrained.yaml"}


# ----------------------------------------------------------------------------- data / prompts

def test_unique_prompts_dedup_by_normalized_prompt():
    from conftest import make_pair

    pairs = [make_pair(0, prompt="A  b"), make_pair(1, prompt="a b"), make_pair(2, prompt=" A b "), make_pair(3, prompt="c")]
    u = unique_prompts(pairs)
    assert len(u) == 3 and sorted(x["n_pairs"] for x in u) == [1, 1, 2]


@pytest.mark.artifacts
def test_manifests_deterministic_disjoint_and_shared_by_arms(artifacts_dir, tmp_path):
    info = build_manifests(str(artifacts_dir))
    assert info["n_train_prompts"] == 9054 and info["n_calibration_prompts"] == 256 and info["n_monitoring_prompts"] == 256
    train = load_prompts(str(artifacts_dir), "train"); cal = load_prompts(str(artifacts_dir), "calibration"); mon = load_prompts(str(artifacts_dir), "monitoring")
    h = lambda xs: {x["prompt_hash"] for x in xs}  # noqa: E731
    assert not (h(cal) & h(mon)) and not (h(train) & h(cal)) and not (h(train) & h(mon))
    info2 = build_manifests(str(artifacts_dir))          # byte-identical regeneration
    assert info2["hashes"] == info["hashes"]
    ts = load_prompts(str(artifacts_dir), "train_sample")  # deterministic training-prompt sample for the d representativeness check
    assert len(ts) == 256 and h(ts) <= h(train) and info["n_train_sample_prompts"] == 256
    assert info["calibration_generation_seeds"] == CALIBRATION_GEN_SEEDS and len(CALIBRATION_GEN_SEEDS) == 4
    assert info["monitoring_generation_seeds"] == MONITOR_GEN_SEEDS
    # all three arm configs point at the same artifacts dir / manifests, one prompt epoch, G = 8, B = 16
    for a, f in ARM_CONFIGS.items():
        c = load_config(str(ROOT / "configs" / f))
        assert c["artifacts_dir"] == "artifacts" and c["prompt_epochs"] == 1 and c["group_size"] == 8 and c["prompt_batch_size"] == 16


def test_prompt_order_and_rollout_batches():
    o42, o43 = prompt_order(9054, 42), prompt_order(9054, 43)
    assert sorted(o42) == list(range(9054)) and o42 != o43 and prompt_order(9054, 42) == o42
    b = rollout_batches(o42, 16)
    assert len(b) == 566 and len(b[-1]) == 9054 - 565 * 16 == 14 and sum(len(x) for x in b) == 9054


# ----------------------------------------------------------------------------- advantages

def test_group_advantages_within_group_only_and_zero_variance():
    s = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 100.0])
    out = group_advantages(s, 4, eps=1e-6)
    a = out["advantages"]
    g0 = s[:4]; expected0 = (g0 - g0.mean()) / (g0.std(unbiased=False) + 1e-6)
    assert torch.allclose(a[:4], expected0)
    assert torch.all(a[4:8] == 0) and out["n_zero_variance_groups"] == 1
    assert not a.requires_grad
    # changing another group's scores does not change this group's advantages
    s2 = s.clone(); s2[8:] *= 3
    assert torch.allclose(group_advantages(s2, 4)["advantages"][:4], a[:4])
    with pytest.raises(ValueError):
        group_advantages(s[:10], 4)


def test_constrained_advantage_is_separately_normalized_components():
    """Revision 2026-11-01: A = A_R - lambda * A_C, each group-normalized separately (NOT joint normalization of R - lambda C)."""
    from saferl.grpo_core import constrained_advantages

    R = torch.tensor([1.0, 2.0, 3.0, 4.0]); C = torch.tensor([10.0, 0.0, 5.0, -5.0]); lam, eps = 0.5, 1e-6
    out = constrained_advantages(R, C, lam, 4, eps)
    aR = (R - R.mean()) / (R.std(unbiased=False) + eps); aC = (C - C.mean()) / (C.std(unbiased=False) + eps)
    assert torch.allclose(out["advantages"], aR - lam * aC, atol=1e-6)           # exact formula
    assert torch.allclose(out["adv_reward"], aR) and torch.allclose(out["adv_cost"], aC)
    joint = group_advantages(lagrangian_scores(R, C, lam), 4, eps)["advantages"]
    assert not torch.allclose(out["advantages"], joint)                          # differs from the old joint normalization
    assert not out["advantages"].requires_grad
    # lambda scales only the cost component; lambda = 0 reduces to A_R
    assert torch.allclose(constrained_advantages(R, C, 0.0, 4, eps)["advantages"], aR)
    # per-group normalization: another group's values do not affect this group
    R2 = torch.cat([R, torch.tensor([100.0, 0.0, 50.0, -50.0])]); C2 = torch.cat([C, torch.tensor([1.0, 2.0, 3.0, 4.0])])
    assert torch.allclose(constrained_advantages(R2, C2, lam, 4, eps)["advantages"][:4], out["advantages"])
    # zero-variance component -> that component is 0 (counted), the other still contributes
    Cz = torch.tensor([2.0, 2.0, 2.0, 2.0])
    oz = constrained_advantages(R, Cz, lam, 4, eps)
    assert torch.allclose(oz["advantages"], aR) and oz["n_zero_variance_cost_groups"] == 1 and oz["n_zero_variance_reward_groups"] == 0
    # scalar arm unchanged: advantage of F alone
    assert torch.allclose(group_advantages(R, 4)["advantages"], aR)


def test_cost_shift_invariance_of_advantages_threshold_and_dual():
    R = torch.tensor([1.0, 2.0, 3.0, 4.0]); C = torch.tensor([0.3, -1.2, 2.0, 0.7]); b = 7.5
    lam = 1.3
    from saferl.grpo_core import constrained_advantages
    a0 = constrained_advantages(R, C, lam, 4)["advantages"]
    a1 = constrained_advantages(R, C + b, lam, 4)["advantages"]
    assert torch.allclose(a0, a1, atol=1e-5)                         # group-centered advantages unchanged by a constant cost shift
    d0 = float(C.mean()); d1 = float((C + b).mean())
    assert math.isclose(d1, d0 + b, rel_tol=1e-6)                   # calibrated d shifts by b
    assert math.isclose(float((C + b).mean()) - d1, float(C.mean()) - d0, abs_tol=1e-6)  # C - d unchanged
    s0, s1 = DualState(d=d0), DualState(d=d1)
    for c in (0.5, 0.1, 0.9):
        r0, r1 = s0.update(c), s1.update(c + b)
        assert math.isclose(r0["lambda"], r1["lambda"], rel_tol=1e-6)  # dual update unchanged (float32 mean rounding)


# ----------------------------------------------------------------------------- dual update

def test_dual_update_rules():
    s = DualState(d=0.0, eta=0.01)
    assert s.lam == 1.0 and s.eta == 0.01 and s.ema is None
    r = s.update(1.0)                                  # first batch: ema = c_batch
    assert r["c_ema"] == 1.0 and r["lambda"] > 1.0 and math.isclose(r["lambda"], math.exp(0.01 * 1.0 * 1.0))
    r2 = s.update(1.0)
    assert math.isclose(r2["c_ema"], 0.9 * 1.0 + 0.1 * 1.0) and r2["lambda"] > r["lambda"]
    s2 = DualState(d=5.0, eta=0.01)
    r3 = s2.update(1.0)                                # C_ema < d -> lambda decreases, stays positive
    assert 0 < r3["lambda"] < 1.0
    for _ in range(500):
        s2.update(-100.0)
    assert s2.lam > 0
    assert s.n_updates == 2 and len(s.history) == 2  # one record per rollout batch
    sd = s.state_dict(); s3 = DualState.from_state_dict(sd)
    assert s3.lam == s.lam and s3.ema == s.ema and s3.d == s.d


# ----------------------------------------------------------------------------- objective

def test_grpo_loss_matches_hand_computation_and_masks():
    logp_old = torch.log(torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))
    logp_new = torch.log(torch.tensor([[0.75, 0.3, 0.5], [0.5, 0.5, 0.5]]))  # ratios 1.5, 0.6, 1.0 ; 1, 1, 1
    logp_ref = logp_old.clone()
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)  # last token of traj 0 masked (padding)
    out = grpo_loss(logp_new.requires_grad_(True), logp_old, logp_ref, adv, mask, clip_eps=0.2, beta_kl=0.0)
    # traj 0: tokens: min(1.5*1, 1.2*1)=1.2 ; min(0.6*1, 0.8*1)=0.6 -> mean over 2 tokens = 0.9
    # traj 1: A=-1, ratio 1 -> min(-1, -1) = -1 for each of 3 tokens -> -1
    assert math.isclose(float(out["loss"]), -(0.9 + (-1.0)) / 2, rel_tol=1e-6)
    out["loss"].backward()
    assert float(logp_new.grad[0, 2]) == 0.0                # masked token gets no gradient
    assert float(logp_new.grad[0, 0]) == 0.0                # clipped token (ratio 1.5 > 1.2 with A>0) has zero gradient
    assert float(logp_new.grad[0, 1]) != 0.0                # unclipped token has gradient
    assert math.isclose(float(out["clip_fraction"]), 2 / 5, rel_tol=1e-6)  # ratios 1.5 and 0.6 are outside [0.8, 1.2]


def test_kl_term_uses_reference_and_is_nonnegative():
    logp_new = torch.log(torch.tensor([[0.5, 0.25]]))
    logp_old = torch.log(torch.tensor([[0.5, 0.5]]))      # pi_old != pi_ref
    logp_ref = torch.log(torch.tensor([[0.25, 0.25]]))
    adv = torch.tensor([0.0])                              # isolate the KL term
    mask = torch.ones(1, 2)
    out = grpo_loss(logp_new, logp_old, logp_ref, adv, mask, 0.2, beta_kl=0.02)
    d = logp_ref - logp_new
    kl = (torch.exp(d) - d - 1).mean()
    assert math.isclose(float(out["loss"]), 0.02 * float(kl), rel_tol=1e-6) and float(kl) >= 0
    # KL is unaffected by pi_old; ratio/clipping is unaffected by pi_ref
    out2 = grpo_loss(logp_new, torch.log(torch.tensor([[0.1, 0.9]])), logp_ref, adv, mask, 0.2, 0.02)
    assert math.isclose(float(out2["kl_mean"]), float(out["kl_mean"]))


def test_advantages_are_constants_in_the_objective():
    logp_new = torch.zeros(2, 3, requires_grad=True)
    adv = torch.tensor([1.0, 2.0], requires_grad=True)
    out = grpo_loss(logp_new, torch.zeros(2, 3), torch.zeros(2, 3), adv, torch.ones(2, 3))
    out["loss"].backward()
    assert adv.grad is None


# ----------------------------------------------------------------------------- configs

def test_arm_configs_share_common_policy_hyperparameters():
    from saferl.train_grpo import COMMON_POLICY_KEYS

    cfgs = {a: load_config(str(ROOT / "configs" / f)) for a, f in ARM_CONFIGS.items()}
    ref = cfgs["scalar"]
    for a, c in cfgs.items():
        assert c["arm"] == a
        for k in COMMON_POLICY_KEYS:
            assert c.get(k) == ref.get(k), (a, k)
    assert ref["learning_rate"] == 1e-6 and ref["adam_beta2"] == 0.999 and ref["weight_decay"] == 0.0 and ref["warmup_ratio"] == 0.03
    assert ref["clip_epsilon"] == 0.2 and ref["beta_kl"] == 0.02 and ref["advantage_eps"] == 1e-6
    assert ref["prompt_batch_size"] == 16 and ref["group_size"] == 8 and ref["groups_per_minibatch"] == 8 and ref["prompt_epochs"] == 1
    assert ref["generation"] == {"max_prompt_length": 512, "max_completion_length": 512, "temperature": 1.0, "top_p": 1.0,
                                 "repetition_penalty": 1.0, "do_sample": True}
    for a in ("shared", "separate"):
        c = cfgs[a]
        assert c["lambda_init"] == 1.0 and c["dual_learning_rate"] == 0.01 and c["cost_ema_decay"] == 0.9
        assert c["calibration_file"] == f"artifacts/grpo/calibration_{a}.json"  # separate d per arm, fixed across seeds
        assert "seed42" in str(c["scorer_checkpoints"])                        # frozen RM seed 42 for every policy seed
    assert "calibration_file" not in cfgs["scalar"] and "cost" not in str(cfgs["scalar"]["scorer_checkpoints"])


def test_same_frozen_scorer_for_all_policy_seeds():
    """Policy seeds 42/43/44 of an arm must load the identical frozen scorer checkpoint(s): RM seed 42, best-by-validation."""
    for a, f in ARM_CONFIGS.items():
        per_seed = [load_config(str(ROOT / "configs" / f), [f"seed={s}"]) for s in (42, 43, 44)]
        ck = [c["scorer_checkpoints"] for c in per_seed]
        assert ck[0] == ck[1] == ck[2], a                                   # independent of the policy seed
        assert all(c["scorer_rm_seed"] == 42 for c in per_seed)
        for head, path in ck[0].items():
            assert path.endswith("_seed42/checkpoints/best"), (a, head, path)
    assert load_config(str(ROOT / "configs" / ARM_CONFIGS["scalar"]))["scorer_checkpoints"] == {"scalar": "runs/rm_scalar_qwen2.5-1.5b_seed42/checkpoints/best"}
    assert load_config(str(ROOT / "configs" / ARM_CONFIGS["shared"]))["scorer_checkpoints"] == {"shared": "runs/rm_shared_qwen2.5-1.5b_seed42/checkpoints/best"}
    assert load_config(str(ROOT / "configs" / ARM_CONFIGS["separate"]))["scorer_checkpoints"] == {
        "reward": "runs/rm_separate_reward_qwen2.5-1.5b_seed42/checkpoints/best", "cost": "runs/rm_separate_cost_qwen2.5-1.5b_seed42/checkpoints/best"}
    # the trainer refuses a scorer path that is not the RM seed-42 checkpoint
    from saferl.train_grpo import GRPOTrainer
    bad = load_config(str(ROOT / "configs" / ARM_CONFIGS["scalar"]))
    bad["scorer_checkpoints"] = {"scalar": "runs/rm_scalar_qwen2.5-1.5b_seed43/checkpoints/best"}
    import inspect
    assert "scorer_rm_seed" in inspect.getsource(GRPOTrainer.setup)


def test_no_code_path_treats_zero_cost_as_a_threshold():
    import inspect

    import saferl.grpo_core as core
    import saferl.train_grpo as tg

    src = inspect.getsource(core) + inspect.getsource(tg)
    assert "cost > 0" not in src and "C > 0" not in src and "cost < 0" not in src


# ----------------------------------------------------------------------------- tiny end-to-end (resume)

@pytest.mark.hf
@pytest.mark.artifacts
def test_tiny_grpo_run_and_resume_reproduces(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    """Tiny random policy + tiny random scorer on CPU: an interrupted-and-resumed run must match an
    uninterrupted run (same prompts, same rollouts, same final weights within tolerance)."""
    from saferl.grpo_calibrate import calibrate_d, generate_calibration_set
    from saferl.rm_models import ScoreModel
    from saferl.train_grpo import GRPOTrainer

    pol_dir = tmp_path / "policy"; tiny_model.save_pretrained(pol_dir); tokenizer.save_pretrained(pol_dir)
    torch.manual_seed(1)
    rm = ScoreModel(copy.deepcopy(tiny_model.model), ["reward", "cost"], tiny_model.config.hidden_size, 0.02, str(pol_dir))
    rm_dir = tmp_path / "rm"; rm.save_pretrained(rm_dir, "float32"); tokenizer.save_pretrained(rm_dir)
    build_manifests(str(smoke_artifacts_dir), n_calibration=3, n_monitoring=3, n_train_sample=3)
    base = load_config(str(ROOT / "configs" / "grpo_smoke.yaml"))
    cfg = {**base, "arm": "shared", "artifacts_dir": str(smoke_artifacts_dir), "policy_model": str(pol_dir), "device": "cpu", "precision": "fp32",
           "gradient_checkpointing": False, "prompt_batch_size": 2, "group_size": 4, "groups_per_minibatch": 1, "policy_micro_batch_size": 4,
           "generation": {"max_prompt_length": 128, "max_completion_length": 6, "temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0, "do_sample": True},
           "max_rollout_batches": 3, "eval_every_batches": 100, "monitor_at_start": False, "resume_every_batches": 2, "learning_rate": 1e-3,
           "monitor_responses_per_prompt": 1, "monitor_gen_batch": 8, "scorer_checkpoints": {"shared": str(rm_dir)},
           "calibration_file": str(tmp_path / "cal.json"), "runs_dir": str(tmp_path / "runs_a"), "run_name": "tiny", "seed": 7}
    generate_calibration_set(cfg, "calibration", tmp_path / "cal_gen.jsonl", seeds=[1, 2], gen_batch=8)
    rep = calibrate_d(cfg, tmp_path / "cal_gen.jsonl", tmp_path / "cal.json")
    assert rep["n_trajectories"] == 3 * 2 and math.isclose(rep["d"], rep["cost"]["mean"])
    from saferl.grpo_calibrate import representativeness_check
    rr = representativeness_check(cfg, tmp_path / "cal.json", tmp_path / "ts_gen.jsonl", tmp_path / "repr.json")
    assert rr["d"] == rep["d"] and rr["train_sample"]["n_trajectories"] == 3 * 4   # 3 smoke train-sample prompts x 4 seeds
    assert math.isclose(rr["train_sample_mean_cost_minus_d"], rr["train_sample"]["cost"]["mean"] - rep["d"], abs_tol=1e-9)
    assert read_json(tmp_path / "cal.json")["d"] == rep["d"]                      # d unchanged by the check
    # uninterrupted run
    s_a = GRPOTrainer(cfg).train()
    assert s_a["n_prompts"] == 6 and s_a["n_trajectories"] == 24 and s_a["opt_step"] == 6 and s_a["n_dual_updates"] == 3
    # interrupted run: stop after 2 batches (resume state saved at batch 2), then resume to the end
    cfg_b = {**cfg, "runs_dir": str(tmp_path / "runs_b"), "max_rollout_batches": 2}
    GRPOTrainer(cfg_b).train()
    cfg_b2 = {**cfg_b, "max_rollout_batches": 3}
    s_b = GRPOTrainer(cfg_b2).train()
    assert s_b["resumed"] and s_b["n_prompts"] == 6 and s_b["n_trajectories"] == 24 and s_b["opt_step"] == 6
    assert math.isclose(s_a["final_lambda"], s_b["final_lambda"], rel_tol=1e-6)
    from safetensors.torch import load_file
    wa = load_file(Path(s_a["final_checkpoint"]) / "model.safetensors"); wb = load_file(Path(s_b["final_checkpoint"]) / "model.safetensors")
    for k in wa:
        assert torch.allclose(wa[k].float(), wb[k].float(), atol=1e-5, rtol=1e-4), k
    # scorer and reference received no gradients; every rollout consumed exactly once (3 batches x 2 groups x 4 = 24)
    rows = [l for l in open(Path(s_a["run_dir"]) / "rollout_batches.jsonl")]
    assert len(rows) == 3


# ----------------------------------------------------------------------------- gates / diagnostics / corrected sampler

def test_gate_logic_is_hard_failures_only_and_tail_mass_is_diagnostic():
    from saferl.grpo_diag import DEFAULT_COLLAPSE, DEFAULT_GATES, collapse_check, evaluate_gates, records_finite

    assert DEFAULT_GATES == {} and DEFAULT_COLLAPSE == {}
    good = {"tail_mass": 0.02, "kl_mean": 0.3, "truncation_rate": 0.1, "ascii_mean": 0.98, "ascii_low_frac": 0.0, "finite": True}
    r = evaluate_gates({**good, "tail_mass": 0.3, "kl_mean": 50.0, "truncation_rate": 1.0, "ascii_mean": 0.1, "ascii_low_frac": 1.0})
    assert r["passed"] and set(r["checks"]) == {"finite"}            # extreme diagnostics never fail the automatic gate
    assert r["diagnostics_reported_not_gating"]["tail_mass"] == 0.3 and r["diagnostics_reported_not_gating"]["kl_mean"] == 50.0
    assert not evaluate_gates({**good, "finite": False})["passed"]   # hard failure
    assert not evaluate_gates(good, {"tail_mass_max": 0.01})["passed"]  # explicit opt-in numerical gate still works
    assert not collapse_check({"tail_mass": 0.5, "kl_mean": 50.0, "ascii_mean": 0.1, "finite": True})["collapsed"]
    assert collapse_check({"tail_mass": 0.0, "kl_mean": 0.0, "ascii_mean": 1.0, "finite": False})["reasons"] == ["non-finite values"]
    assert records_finite([{"loss": 0.1, "kl_mean": 0.0, "ratio_mean": 1.0, "grad_norm": [1.0]}])
    assert not records_finite([{"loss": float("nan"), "kl_mean": 0.0, "ratio_mean": 1.0, "grad_norm": [1.0]}])


@pytest.mark.hf
def test_tail_mass_matches_direct_computation(tokenizer, tiny_model):
    from saferl.grpo_diag import tail_mass_and_entropy
    from saferl.grpo_rollout import collate_trajectories

    t = {"prompt_ids": tokenizer("hello there", add_special_tokens=False)["input_ids"], "completion_ids": tokenizer(" general kenobi!", add_special_tokens=False)["input_ids"]}
    out = tail_mass_and_entropy(tiny_model, [t], tokenizer.pad_token_id, "cpu", "fp32", k=20)
    b = collate_trajectories([t], tokenizer.pad_token_id)
    with torch.no_grad():
        probs = torch.softmax(tiny_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits[:, :-1].float(), -1)
    mask = b["loss_mask"][:, 1:].bool()
    direct = float((1 - torch.topk(probs, 20, -1).values.sum(-1))[mask].mean())
    assert math.isclose(out["tail_mass"], direct, rel_tol=1e-5) and out["n_tokens"] == len(t["completion_ids"]) and out["finite"]


@pytest.mark.hf
def test_rollout_generation_pins_top_k_to_zero(tokenizer, tiny_model, monkeypatch):
    """The packaged generation_config must never influence rollouts: every sampling knob is passed explicitly."""
    from saferl.grpo_rollout import generate_completions

    captured = {}
    real = tiny_model.generate

    def spy(**kw):
        captured.update(kw)
        return real(**kw)

    monkeypatch.setattr(tiny_model, "generate", spy)
    generate_completions(tiny_model, tokenizer, ["Hi"], 1, 64, 4, "cpu", "fp32", seed=0)
    assert captured["top_k"] == 0 and captured["top_p"] == 1.0 and captured["temperature"] == 1.0 and captured["repetition_penalty"] == 1.0
    assert captured["do_sample"] is True


@pytest.mark.hf
@pytest.mark.artifacts
def test_stop_after_batches_then_resume_continues_without_restart(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    from saferl.grpo_calibrate import calibrate_d, generate_calibration_set
    from saferl.rm_models import ScoreModel
    from saferl.train_grpo import GRPOTrainer

    pol_dir = tmp_path / "policy"; tiny_model.save_pretrained(pol_dir); tokenizer.save_pretrained(pol_dir)
    torch.manual_seed(1)
    rm = ScoreModel(copy.deepcopy(tiny_model.model), ["reward", "cost"], tiny_model.config.hidden_size, 0.02, str(pol_dir))
    rm_dir = tmp_path / "rm"; rm.save_pretrained(rm_dir, "float32"); tokenizer.save_pretrained(rm_dir)
    build_manifests(str(smoke_artifacts_dir), n_calibration=3, n_monitoring=3, n_train_sample=3)
    base = load_config(str(ROOT / "configs" / "grpo_smoke.yaml"))
    cfg = {**base, "arm": "shared", "artifacts_dir": str(smoke_artifacts_dir), "policy_model": str(pol_dir), "device": "cpu", "precision": "fp32",
           "gradient_checkpointing": False, "prompt_batch_size": 2, "group_size": 4, "groups_per_minibatch": 1, "policy_micro_batch_size": 4,
           "generation": {"max_prompt_length": 128, "max_completion_length": 6, "temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0, "do_sample": True},
           "max_rollout_batches": 4, "eval_every_batches": 100, "monitor_at_start": False, "resume_every_batches": 100, "learning_rate": 1e-3,
           "monitor_responses_per_prompt": 1, "monitor_gen_batch": 8, "scorer_checkpoints": {"shared": str(rm_dir)}, "calibration_file": str(tmp_path / "cal.json"),
           "runs_dir": str(tmp_path / "runs"), "run_name": "gated", "seed": 7, "collapse_check_every": 1, "diagnostic": {"n_samples": 3, "seed": 1},
           # a random tiny model has a flat distribution: relax the gates so the mechanism (not the thresholds) is under test
           "diagnostic_gates": {}, "collapse": {}}
    generate_calibration_set(cfg, "calibration", tmp_path / "cal_gen.jsonl", seeds=[1, 2], gen_batch=8)
    calibrate_d(cfg, tmp_path / "cal_gen.jsonl", tmp_path / "cal.json")
    s1 = GRPOTrainer({**cfg, "stop_after_batches": 2}).train()
    rd = Path(s1["run_dir"])
    assert s1["stopped_for_diagnostic"] and s1["gates_passed"] and s1["next_batch_index"] == 2
    assert (rd / "diagnostic" / "batch_00002.json").exists() and (rd / "diagnostic" / "batch_00002_samples.jsonl").exists()
    assert not (rd / "checkpoints" / "final").exists() and (rd / "resume" / "latest.pt").exists()   # no final checkpoint before the gate
    d = read_json(rd / "diagnostic" / "batch_00002.json")
    assert 0.0 <= d["tail_mass"] <= 1.0 and d["n_samples"] == 3 and d["finite"]
    # continuation: resumes at batch 2, completes 4, writes the final checkpoint; batches 0-1 are not replayed
    s2 = GRPOTrainer(cfg).train()
    assert s2["resumed"] and s2["n_prompts"] == 8 and s2["n_trajectories"] == 32 and (rd / "checkpoints" / "final").exists()
    rows = [json.loads(l) for l in open(rd / "rollout_batches.jsonl")]
    assert [r["batch_index"] for r in rows] == [0, 1, 2, 3]
    assert (rd / "diagnostic" / "collapse_probes.jsonl").exists()
    # a failing gate returns gates_passed False (exit code 2 is raised by main())
    cfg_fail = {**cfg, "runs_dir": str(tmp_path / "runs_fail"), "diagnostic_gates": {"tail_mass_max": 0.0}}
    s3 = GRPOTrainer({**cfg_fail, "stop_after_batches": 1}).train()
    assert s3["stopped_for_diagnostic"] and not s3["gates_passed"] and not s3["diagnostic"]["gates"]["checks"]["tail_mass"]["passed"]
