"""Six-arm study: frozen configs, prompt exposure, thresholds, fixed-budget checkpoint rule, paired bootstrap, raw-score REBEL with
GRPO's dual convention (tiny CPU run), and GRPO's explicit checkpoint list (default behaviour unchanged)."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from saferl.utils import load_config, read_json, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["arm0_rebel_separate_d1", "arm1_rebel_separate_d0", "arm2_rebel_scalar", "arm3_grpo_separate_d1", "arm4_grpo_separate_d0", "arm5_grpo_scalar"]


def test_six_arm_configs_are_frozen_and_matched():
    cfgs = {a: load_config(str(ROOT / "configs" / "sixarm" / f"{a}.yaml")) for a in ARMS}
    gen = {json.dumps(c["generation"], sort_keys=True) for c in cfgs.values()}
    assert len(gen) == 1 and cfgs[ARMS[0]]["generation"]["temperature"] == 1.0 and cfgs[ARMS[0]]["generation"]["top_p"] == 1.0
    for a, c in cfgs.items():
        assert c["seed"] == 42 and c["max_rollout_batches"] == 200 and c["checkpoint_at_batches"] == [100, 200] and c["artifacts_dir"] == "artifacts/sixarm"
        assert c["prompt_batch_size"] == 16 and c["group_size"] == 8 and c["policy_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert c["scorer_rm_seed"] == 42 and all("_seed42/" in p for p in c["scorer_checkpoints"].values())
        assert c["lambda_init"] == 1.0 and c["dual_learning_rate"] == 0.01 and c["cost_ema_decay"] == 0.9     # same dual hyperparameters everywhere
        if "rebel" in a:
            assert c["score_units"] == "raw" and c["dual_rule"] == "grpo_log_lambda" and c["pairs_per_minibatch"] == 32 and c["inner_epochs"] == 1
            assert c["rebel_eta"] == 1.0 and c["kl_to_initial_coef"] == 0
        else:
            g = load_config(str(ROOT / "configs" / "grpo_common.yaml"))
            for k in ("clip_epsilon", "beta_kl", "advantage_eps", "groups_per_minibatch", "learning_rate", "lr_scheduler", "warmup_ratio", "max_grad_norm", "adam_beta2"):
                assert c[k] == g[k], k                                                                            # GRPO untouched
    sep = {"reward": "runs/rm_separate_reward_qwen2.5-1.5b_seed42/checkpoints/best", "cost": "runs/rm_separate_cost_qwen2.5-1.5b_seed42/checkpoints/best"}
    for a in ("arm0_rebel_separate_d1", "arm1_rebel_separate_d0", "arm3_grpo_separate_d1", "arm4_grpo_separate_d0"):
        assert cfgs[a]["arm"] == "separate" and cfgs[a]["scorer_checkpoints"] == sep
    assert cfgs["arm0_rebel_separate_d1"]["calibration_file"].endswith("_d1.json") and cfgs["arm3_grpo_separate_d1"]["calibration_file"].endswith("_d1.json")
    assert cfgs["arm1_rebel_separate_d0"]["calibration_file"].endswith("_d0.json") and cfgs["arm4_grpo_separate_d0"]["calibration_file"].endswith("_d0.json")
    for a in ("arm2_rebel_scalar", "arm5_grpo_scalar"):
        assert cfgs[a]["arm"] == "scalar" and cfgs[a]["calibration_file"] is None
    ev = load_config(str(ROOT / "configs" / "eval_sixarm.yaml"))
    assert set(["base"] + [f"sixarm_arm{k}" for k in range(6)]) <= set(ev["methods"])   # (older methods are inherited; the merge is restricted with --methods)
    assert all(ev["methods"][f"sixarm_arm{k}"]["checkpoint_rule"] == "after_batches:200" for k in range(6))
    assert ev["paired_comparisons"] == [["sixarm_arm0", "sixarm_arm3"], ["sixarm_arm1", "sixarm_arm4"], ["sixarm_arm2", "sixarm_arm5"], ["sixarm_arm0", "sixarm_arm1"], ["sixarm_arm3", "sixarm_arm4"]]
    assert ev["generation"] == load_config(str(ROOT / "configs" / "eval.yaml"))["generation"]


@pytest.mark.artifacts
def test_prompt_exposure_is_identical_for_both_trainers_and_leak_free():
    info_path = ROOT / "artifacts" / "sixarm" / "prompt_exposure.json"
    if not info_path.exists():
        pytest.skip("artifacts/sixarm not built")
    from saferl.grpo_data import load_prompts, prompt_order, rollout_batches
    from saferl.rebel_data import load_rebel_prompts
    info = read_json(info_path)
    h = info["ordered_prompt_hashes"]
    assert len(h) == 3200 == len(set(h)) and all(v == 0 for v in info["hash_intersections"].values())
    g = load_prompts(str(ROOT / "artifacts" / "sixarm"), "train"); r = load_rebel_prompts(str(ROOT / "artifacts" / "sixarm"), "train")
    assert [p["prompt_hash"] for p in g] == [p["prompt_hash"] for p in r]
    order = prompt_order(len(g), 42); first = [g[i]["prompt_hash"] for b in rollout_batches(order, 16)[:200] for i in b]
    assert first == h
    hp = read_json(ROOT / "artifacts" / "sixarm_hpval" / "hpval_info.json")
    assert hp["pilot_training_prompts"] == 640 and all(v == 0 for v in hp["hash_intersections"].values())
    assert not (set(hp["prompt_hashes"]) & {p["prompt_hash"] for p in g})          # never in the MAIN manifest (no resumption can reach them)
    mon = {p["prompt_hash"] for p in load_rebel_prompts(str(ROOT / "artifacts" / "sixarm"), "monitoring")}
    assert not (set(hp["prompt_hashes"]) & mon) and not (mon & {p["prompt_hash"] for p in g})   # the development set is trained on by nothing


def test_thresholds_derived_from_calibration(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from sixarm_calibration import derive
    rng = np.random.default_rng(0)
    rows = [{"prompt_hash": f"p{i // 4:03d}", "k": i % 4, "cost": float(rng.normal(-4.0, 1.5)), "reward": 0.0} for i in range(64)]
    c = np.array([r["cost"] for r in rows])
    cal = {"arm": "separate", "d": float(c.mean()), "cost": {"mean": float(c.mean()), "std": float(c.std())}, "scorer_checkpoints": {"cost": {"model.safetensors": "x"}},
           "n_trajectories": 64, "generations_sha256": "abc"}
    write_json(tmp_path / "calibration_separate.json", cal); write_jsonl(tmp_path / "calibration_separate.scores.jsonl", rows)
    out = derive(tmp_path / "calibration_separate.json", tmp_path / "out", n_boot=200)
    d0 = read_json(tmp_path / "out" / "calibration_separate_d0.json"); d1 = read_json(tmp_path / "out" / "calibration_separate_d1.json")
    assert math.isclose(d0["d"], c.mean()) and math.isclose(d1["d"], c.mean() - 0.5 * c.std()) and d0["arm"] == "separate"
    assert d1["uncertainty"]["d_se"] > 0 and 0 <= d1["feasibility_under_pi0"]["frac_responses_at_or_below_d"] <= 1
    assert d0["scorer_checkpoints"] == cal["scorer_checkpoints"] and out["d1"]["d"] < out["d0"]["d"]
    bad = {**cal, "arm": "shared"}; write_json(tmp_path / "bad.json", bad); write_jsonl(tmp_path / "bad.scores.jsonl", rows)
    with pytest.raises(ValueError):
        derive(tmp_path / "bad.json", tmp_path / "out2", n_boot=10)


def test_after_batches_rule(tmp_path):
    from saferl.eval_common import verify_checkpoint_rule
    ok = tmp_path / "batch_00200"; ok.mkdir(); write_json(ok / "trainer_state.json", {"intermediate": True, "batch_index": 200, "next_batch_index": 200, "n_prompts": 3200, "algorithm": "rebel"})
    assert verify_checkpoint_rule(str(ok), "after_batches:200")["trainer_state"]["n_prompts"] == 3200
    for rule in ("after_batches:100", "rebel_final_1_prompt_epoch", "grpo_final_1_prompt_epoch"):
        with pytest.raises(ValueError):
            verify_checkpoint_rule(str(ok), rule)
    with pytest.raises(ValueError):
        verify_checkpoint_rule(str(tmp_path / "best"), "after_batches:200")


def test_paired_bootstrap_on_synthetic_scores(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_pairs import paired
    rng = np.random.default_rng(1)
    a = {f"h{i}": {"R": float(rng.normal(0, 1)), "C": float(rng.normal(0, 1))} for i in range(500)}
    b = {h: {"R": v["R"] - 0.3, "C": v["C"] + 0.5} for h, v in a.items()}                     # constant per-prompt offsets
    r = paired(a, b, 500, 0)
    assert math.isclose(r["helpfulness"]["point"], 0.3, abs_tol=1e-9) and math.isclose(r["harmfulness"]["point"], -0.5, abs_tol=1e-9)
    assert r["helpfulness"]["ci_high"] - r["helpfulness"]["ci_low"] < 1e-6 and r["helpfulness"]["frac_prompts_a_higher"] == 1.0
    with pytest.raises(ValueError):
        paired(a, {k: v for k, v in list(b.items())[:10]}, 10, 0)


# ----------------------------------------------------------------------------- tiny CPU runs

def _tiny(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    from conftest import synthetic_pool
    from saferl.rebel_calibrate import calibrate_arm, generate_calibration
    from saferl.rebel_data import build_rebel_manifests
    from saferl.rm_models import ScoreModel
    import shutil

    art = tmp_path / "art"; shutil.copytree(smoke_artifacts_dir, art)
    build_rebel_manifests(str(art), synthetic_pool(200, "train", seed=1, n_prompts=120), synthetic_pool(20, "test", seed=2, n_prompts=10),
                          data_seed=42, n_train=6, n_calibration=3, n_monitoring=4, reference_sources={"Alpaca3-70B": 1})
    for sub in ("grpo",):   # the GRPO trainer reads <artifacts>/grpo/*, the REBEL trainer <artifacts>/rebel/*
        (art / sub).mkdir(exist_ok=True)
        for n in ("train", "calibration", "monitoring"):
            shutil.copyfile(art / "rebel" / f"{n}_prompts.jsonl", art / sub / f"{n}_prompts.jsonl")
    pol_dir = tmp_path / "policy"; tiny_model.save_pretrained(pol_dir); tokenizer.save_pretrained(pol_dir)
    torch.manual_seed(1)
    rm = ScoreModel(copy.deepcopy(tiny_model.model), ["reward", "cost"], tiny_model.config.hidden_size, 0.02, str(pol_dir))
    rm_dir = tmp_path / "rm"; rm.save_pretrained(rm_dir, "float32"); tokenizer.save_pretrained(rm_dir)
    base = load_config(str(ROOT / "configs" / "rebel_smoke.yaml"))
    cfg = {**base, "arm": "shared", "artifacts_dir": str(art), "policy_model": str(pol_dir), "device": "cpu", "precision": "fp32", "gradient_checkpointing": False,
           "attn_implementation": None, "prompt_batch_size": 2, "group_size": 4, "pairs_per_minibatch": 2, "policy_micro_batch_size": 2,
           "generation": {"max_prompt_length": 128, "max_completion_length": 6, "temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0, "do_sample": True},
           "max_rollout_batches": 3, "eval_every_batches": 100, "monitor_at_start": False, "monitor_responses_per_prompt": 1, "monitor_gen_batch": 8,
           "dump_every_batches": 0, "checkpoint_every_batches": 0, "checkpoint_at_batches": [2, 3], "resume_every_batches": 2, "learning_rate": 1e-3, "rebel_eta": 1.0,
           "score_units": "raw", "dual_rule": "grpo_log_lambda", "dual_learning_rate": 0.01, "cost_ema_decay": 0.9, "lambda_init": 1.0,
           "scorer_checkpoints": {"shared": str(rm_dir)}, "runs_dir": str(tmp_path / "runs_a"), "run_name": "tiny", "seed": 7, "collapse_check_every": 0}
    gen = tmp_path / "cal_gen.jsonl"
    generate_calibration(cfg, gen, seeds=[1, 2], gen_batch=8)
    rep = calibrate_arm({**cfg, "d_std": -0.5}, gen, tmp_path / "cal_std.json")
    cal = {"arm": "shared", "d": rep["d_raw"], "d_rule": "mu_C - 0.5 sigma_C", "scorer_checkpoints": rep["scorer_checkpoints"], "generations_sha256": rep["generations_sha256"],
           "cost": {"mean": rep["heads"]["cost"]["mu"], "std": rep["heads"]["cost"]["sigma"]}}
    write_json(tmp_path / "cal_raw.json", cal)
    return {**cfg, "calibration_file": str(tmp_path / "cal_raw.json")}, cal


@pytest.mark.hf
@pytest.mark.artifacts
def test_tiny_rebel_raw_scores_with_grpo_dual_and_fixed_budget_checkpoints(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    from safetensors.torch import load_file
    from saferl.eval_common import verify_checkpoint_rule
    from saferl.train_rebel import REBELTrainer

    cfg, cal = _tiny(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path)
    s = REBELTrainer(cfg).train()
    rd = Path(s["run_dir"])
    rows = [json.loads(l) for l in open(rd / "rollout_batches.jsonl")]
    assert s["n_pairs"] == 12 and s["opt_step"] == 6 and read_json(rd / "config_resolved.json")["score_units"] == "raw"
    ema = None
    for r in rows:   # GRPO's convention, raw units: EMA then mu <- mu + eta * lambda_t * (EMA - d), lambda = exp(mu)
        ema = r["cost_hat"] if ema is None else 0.9 * ema + 0.1 * r["cost_hat"]
        assert r["d"] == cal["d"] and r["dual_rule"] == "grpo_log_lambda" and math.isclose(r["cost_ema"], ema, rel_tol=1e-9)
        assert math.isclose(r["lambda"], r["lambda_used"] * math.exp(0.01 * r["lambda_used"] * (ema - cal["d"])), rel_tol=1e-9)
        assert math.isclose(r["cost_hat"], r["cost_raw_mean"], rel_tol=1e-9)                    # raw, not standardized
        assert math.isclose(r["loss_at_theta_t"], r["target_sq_mean"], rel_tol=1e-6)
    assert rows[1]["lambda_used"] == rows[0]["lambda"]
    assert (rd / "checkpoints" / "batch_00002").exists() and (rd / "checkpoints" / "batch_00003").exists()
    assert verify_checkpoint_rule(str(rd / "checkpoints" / "batch_00003"), "after_batches:3")["trainer_state"]["n_prompts"] == 6
    with pytest.raises(ValueError):
        verify_checkpoint_rule(str(rd / "checkpoints" / "batch_00002"), "after_batches:3")
    # the resume state at batch 2 continues the SAME run: an interrupted run reaches identical weights and lambda
    cfg_b = {**cfg, "runs_dir": str(tmp_path / "runs_b"), "max_rollout_batches": 2}
    REBELTrainer(cfg_b).train()
    s_b = REBELTrainer({**cfg_b, "max_rollout_batches": 3}).train()
    assert s_b["resumed"] and math.isclose(s["final_lambda"], s_b["final_lambda"], rel_tol=1e-9)
    wa = load_file(rd / "checkpoints" / "batch_00003" / "model.safetensors"); wb = load_file(Path(s_b["run_dir"]) / "checkpoints" / "batch_00003" / "model.safetensors")
    assert all(torch.allclose(wa[k].float(), wb[k].float(), atol=1e-5, rtol=1e-4) for k in wa)
    # unconstrained scalar arm on raw scores needs no calibration file
    rm_dir = Path(cfg["scorer_checkpoints"]["shared"])
    from saferl.rm_models import ScoreModel
    sc = ScoreModel(copy.deepcopy(tiny_model.model), ["scalar"], tiny_model.config.hidden_size, 0.02, cfg["policy_model"]); sc_dir = tmp_path / "rm_scalar"
    sc.save_pretrained(sc_dir, "float32"); tokenizer.save_pretrained(sc_dir)
    s2 = REBELTrainer({**cfg, "arm": "scalar", "scorer_checkpoints": {"scalar": str(sc_dir)}, "calibration_file": None, "runs_dir": str(tmp_path / "runs_s"), "max_rollout_batches": 2}).train()
    assert s2["final_lambda"] is None and s2["n_pairs"] == 8


@pytest.mark.hf
@pytest.mark.artifacts
def test_grpo_checkpoint_at_batches_is_additive_and_off_by_default(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    from saferl.eval_common import verify_checkpoint_rule
    from saferl.grpo_calibrate import calibrate_d, generate_calibration_set
    from saferl.grpo_data import build_manifests
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
           "max_rollout_batches": 3, "eval_every_batches": 100, "monitor_at_start": False, "resume_every_batches": 100, "learning_rate": 1e-3,
           "monitor_responses_per_prompt": 1, "monitor_gen_batch": 8, "scorer_checkpoints": {"shared": str(rm_dir)}, "calibration_file": str(tmp_path / "cal.json"),
           "runs_dir": str(tmp_path / "runs_g"), "run_name": "tiny", "seed": 7, "collapse_check_every": 0}
    generate_calibration_set(cfg, "calibration", tmp_path / "cal_gen.jsonl", seeds=[1, 2], gen_batch=8)
    calibrate_d(cfg, tmp_path / "cal_gen.jsonl", tmp_path / "cal.json")
    s = GRPOTrainer({**cfg, "checkpoint_at_batches": [2]}).train()
    rd = Path(s["run_dir"])
    assert (rd / "checkpoints" / "batch_00002").exists() and (rd / "checkpoints" / "final").exists()
    st = read_json(rd / "checkpoints" / "batch_00002" / "trainer_state.json")
    assert st["intermediate"] and st["algorithm"] == "grpo" and st["n_prompts"] == 4
    assert verify_checkpoint_rule(str(rd / "checkpoints" / "batch_00002"), "after_batches:2")["rule"] == "after_batches:2"
    with pytest.raises(ValueError):
        verify_checkpoint_rule(str(rd / "checkpoints" / "batch_00002"), "grpo_final_1_prompt_epoch")      # intermediate refused by the epoch rule
    s0 = GRPOTrainer({**cfg, "runs_dir": str(tmp_path / "runs_g0")}).train()                           # default: no extra checkpoints
    assert sorted(p.name for p in (Path(s0["run_dir"]) / "checkpoints").iterdir()) == ["final"]


# ----------------------------------------------------------------------------- audit additions (KL shaping, EOS identity, resume rows, selection gates)

def test_kl_shaping_units_and_scale():
    from saferl.rebel_core import kl_shaping
    lt = torch.tensor([-300.0, -100.0]); l0 = torch.tensor([-330.0, -110.0]); n = torch.tensor([300, 100])
    s_sum = kl_shaping(lt, l0, n, 0.02, "sequence_sum"); s_tok = kl_shaping(lt, l0, n, 0.02, "token_mean")
    assert torch.allclose(s_sum, torch.tensor([0.6, 0.2])) and torch.allclose(s_tok, torch.tensor([0.002, 0.002]))
    assert torch.equal(kl_shaping(lt, l0, n, 0.0, "token_mean"), torch.zeros(2)) and not s_sum.requires_grad
    with pytest.raises(ValueError):
        kl_shaping(lt, l0, n, 0.02, "per_token")


def test_eos_and_truncation_treatment_is_shared_by_both_trainers_and_the_evaluation():
    import inspect
    import saferl.train_grpo as tg, saferl.train_rebel as tr, saferl.eval_generate as eg, saferl.grpo_scoring as gs
    for mod in (tg, tr):
        src = inspect.getsource(mod)
        assert "generate_completions(" in src and "model.generate(" not in src and "collate_trajectories(" in src   # one sampler, one collation
    assert "eos_ids(" in inspect.getsource(eg) and "truncated = not (comp and comp[-1] in stop)" in inspect.getsource(eg)
    assert '"truncate"' in inspect.getsource(gs)                                                                      # same scorer encoding for every arm
    src_grpo = inspect.getsource(tg.GRPOTrainer.rollout_batch); src_rebel = inspect.getsource(tr.REBELTrainer.rollout_batch)
    assert '"truncation_rate"' in src_grpo and '"truncation_rate"' in src_rebel                                     # logged per batch in both
    assert "length_penalty" not in inspect.getsource(tg) and "length_penalty" not in inspect.getsource(tr)


@pytest.mark.hf
@pytest.mark.artifacts
def test_resume_reproduces_every_logged_quantity_after_the_cut(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    """Optimizer, scheduler, RNG, prompt cursor, lambda and the cost EMA: a run cut after batch 2 and resumed must log batch 2 identically."""
    from saferl.train_rebel import REBELTrainer
    cfg, _ = _tiny(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path)
    cfg = {**cfg, "kl_to_initial_coef": 0.02, "kl_shaping_units": "token_mean"}
    a = REBELTrainer({**cfg, "runs_dir": str(tmp_path / "ra")}).train()
    REBELTrainer({**cfg, "runs_dir": str(tmp_path / "rb"), "max_rollout_batches": 2}).train()
    b = REBELTrainer({**cfg, "runs_dir": str(tmp_path / "rb"), "max_rollout_batches": 3}).train()
    ra = [json.loads(l) for l in open(Path(a["run_dir"]) / "rollout_batches.jsonl")]; rb = [json.loads(l) for l in open(Path(b["run_dir"]) / "rollout_batches.jsonl")]
    assert [r["batch_index"] for r in rb] == [0, 1, 2]
    for key in ("lambda_used", "lambda", "cost_ema", "cost_hat", "loss_at_theta_t", "loss", "lr", "opt_step_total", "reward_raw_mean", "kl_shaping_pair_abs_mean"):
        assert math.isclose(ra[2][key], rb[2][key], rel_tol=1e-5, abs_tol=1e-7), key
    assert ra[2]["kl_shaping_pair_abs_mean"] >= 0 and ra[2]["kl_shaping_units"] == "token_mean" and ra[2]["kl_shaping_coef"] == 0.02
    assert a["opt_step"] == b["opt_step"] == 6 and a["n_prompts"] == b["n_prompts"] == 6


def test_selection_rule_with_dev_gates():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from sixarm_hpval_select import gates, select
    def s(lr, eta, g, arm="scalar", family="rebel", **kw):
        d = {"run": f"sixarm_hpval_{family}_{arm}_lr{lr}_eta{eta}_g{g}_seed42", "family": family, "arm": arm, "lr": lr, "eta": eta, "gamma": g, "finite": True,
             "fit_frac": 0.9, "kl_prev": 1e-4, "clip_frac": 0.001, "dev_scored": True, "trunc_final": 0.1, "kl0_final": 0.05, "dev_help_delta": 0.1,
             "dev_harm_delta": -0.1, "modeled_reward_delta": 0.5}
        d.update(kw); return d
    runs = [s(1e-6, 1.0, 0.0), s(3e-6, 1.0, 0.0), s(1e-5, 1.0, 0.0, kl_prev=5e-3),                      # 1e-5 fails G2
            s(3e-6, 0.5, 0.0, fit_frac=0.92), s(3e-6, 2.0, 0.0, fit_frac=0.99, dev_help_delta=-0.8),      # eta 2 would win on fit but exploits (G5)
            s(3e-6, 1.0, 0.02, kl0_final=0.03), s(3e-6, 1.0, 0.02, arm="separate_d1", kl0_final=0.04)]
    runs += [s(3e-7, 1.0, 0.0, family="grpo"), s(1e-6, 1.0, 0.0, family="grpo"), s(3e-6, 1.0, 0.0, family="grpo", clip_frac=0.2)]   # 3e-6 fails G2
    sel = select(runs)
    r = sel["rebel"]
    assert r["learning_rate"] == 3e-6 and r["rebel_eta"] == 1.0 and r["kl_to_initial_coef"] == 0.02 and r["constrained_check"]["passed"] is True
    assert sel["grpo"]["learning_rate"] == 1e-6
    assert gates(runs[4])["G5_no_exploitation"] is False and gates(runs[2])["G2_kl_prev"] is False
    sel2 = select([s(1e-6, 1.0, 0.0, trunc_final=0.9), s(3e-6, 1.0, 0.0, dev_harm_delta=0.9), s(3e-7, 1.0, 0.0, family="grpo", kl0_final=5.0)])
    assert sel2["rebel"]["learning_rate"] == 1e-6 and "REVIEW" in sel2["rebel"]["reason"] and "REVIEW" in sel2["grpo"]["reason"]


@pytest.mark.hf
@pytest.mark.artifacts
def test_pilot_schedule_spans_the_main_run_and_grpo_monitor_samples_are_opt_in(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path):
    from saferl.train_rebel import REBELTrainer
    cfg, _ = _tiny(tokenizer, tiny_model, smoke_artifacts_dir, tmp_path)
    tr = REBELTrainer({**cfg, "runs_dir": str(tmp_path / "sched"), "schedule_total_batches": 566}); tr.setup()
    assert tr.total_opt_steps == 566 * math.ceil(2 * 2 / 2) and len(tr.batches) == 3            # schedule over 566 batches, only 3 trained
    assert read_json(Path(tr.run_dir) / "config_resolved.json")["total_optimizer_steps"] == tr.total_opt_steps
    tr.batch_log.close()
    src = __import__("inspect").getsource(__import__("saferl.train_grpo", fromlist=["GRPOTrainer"]))
    assert 'cfg.get("monitor_save_samples", False)' in src and 'cfg.get("schedule_total_batches")' in src   # both opt-in, default off
