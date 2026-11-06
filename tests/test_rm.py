"""Reward/cost-model tests: data/manifest reuse, losses, orientation, head sharing/gradients,
parameter independence, label mapping, config consistency, and tiny end-to-end training."""
import copy
import math
from pathlib import Path

import pytest
import torch
import yaml

from conftest import make_pair, synthetic_pool
from saferl import data
from saferl.rm_data import ARCHITECTURES, load_rm_rows, pair_to_rm_rows, pairs_to_rm_rows, safety_sign
from saferl.rm_loss import bt_loss, cost_class_loss, cost_loss
from saferl.utils import load_config, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
ARCH_CONFIGS = {"scalar": "rm_scalar.yaml", "shared": "rm_shared.yaml",
                "separate_reward": "rm_separate_reward.yaml", "separate_cost": "rm_separate_cost.yaml"}


# ----------------------------------------------------------------------------- data / manifests

def test_safety_labels_mapped_from_pku_fields():
    assert safety_sign(True) == -1 and safety_sign(False) == 1
    p = make_pair(0, better=0, safer=1, safe=(False, True))
    h, s = pair_to_rm_rows(p)
    assert s["chosen"] == p["response_1"] and s["chosen_is_safe"] is True and s["chosen_sign"] == -1
    assert s["rejected"] == p["response_0"] and s["rejected_is_safe"] is False and s["rejected_sign"] == 1
    assert h["chosen"] == p["response_0"] and h["chosen_is_safe"] is False and h["rejected_is_safe"] is True


def test_rm_rows_identical_supervision_to_dpo_rows():
    pairs = synthetic_pool(300, "train", seed=11)
    rm = {r["row_id"]: r for r in pairs_to_rm_rows(pairs)}
    dpo = {r["row_id"]: r for r in data.pairs_to_dpo_rows(pairs)}
    assert set(rm) == set(dpo)
    for rid in rm:
        for k in ("prompt", "chosen", "rejected", "objective", "pair_type", "chosen_id"):
            assert rm[rid][k] == dpo[rid][k]


def test_architectures_receive_intended_objectives_and_equal_supervision():
    pairs = synthetic_pool(200, "train", seed=12)
    for arch, spec in ARCHITECTURES.items():
        rows = pairs_to_rm_rows(pairs, spec["objectives"])
        counts = {o: sum(1 for r in rows if r["objective"] == o) for o in spec["objectives"]}
        assert set(counts) == set(spec["objectives"])
        assert all(c == len(pairs) for c in counts.values())
    assert ARCHITECTURES["scalar"]["objectives"] == ["helpfulness", "harmlessness"]  # equal H/S supervision
    assert ARCHITECTURES["separate_reward"]["objectives"] == ["helpfulness"]
    assert ARCHITECTURES["separate_cost"]["objectives"] == ["harmlessness"]
    assert ARCHITECTURES["scalar"]["loss_for"] == {"helpfulness": "bt", "harmlessness": "bt"}  # no pointwise labels


@pytest.mark.artifacts
def test_all_archs_use_exact_dpo_manifests(artifacts_dir):
    dpo = read_jsonl(artifacts_dir / "dpo" / "train.jsonl")
    dpo_ids = {r["row_id"] for r in dpo}
    for arch, spec in ARCHITECTURES.items():
        rows = load_rm_rows(str(artifacts_dir), "train", spec["objectives"])
        ids = {r["row_id"] for r in rows}
        assert ids == {i for i in dpo_ids if i.endswith(":H") and "helpfulness" in spec["objectives"]
                       or i.endswith(":S") and "harmlessness" in spec["objectives"]}
    rows = load_rm_rows(str(artifacts_dir), "train")
    assert len(rows) == 20000 and sum(r["objective"] == "helpfulness" for r in rows) == 10000
    by_id = {r["row_id"]: r for r in rows}
    for r in dpo[:500]:
        assert by_id[r["row_id"]]["chosen"] == r["chosen"] and by_id[r["row_id"]]["rejected"] == r["rejected"]


# ----------------------------------------------------------------------------- losses

def _t(x):
    return torch.tensor([x], dtype=torch.float32)


def test_bt_loss_sign_and_values():
    assert math.isclose(float(bt_loss(_t(0.0), _t(0.0))["loss"]), math.log(2), rel_tol=1e-6)
    assert float(bt_loss(_t(3.0), _t(0.0))["loss"]) < math.log(2) < float(bt_loss(_t(0.0), _t(3.0))["loss"])
    assert math.isclose(float(bt_loss(_t(2.0), _t(-1.0))["loss"]), -math.log(1 / (1 + math.exp(-3.0))), rel_tol=1e-6)


def test_cost_loss_primary_is_pairwise_only_and_ignores_labels():
    # larger C = more harmful: harmful scored above safer -> small loss
    good = cost_loss(_t(-2.0), _t(2.0))
    bad = cost_loss(_t(2.0), _t(-2.0))
    assert float(good["loss"]) < math.log(2) < float(bad["loss"])
    assert math.isclose(float(good["loss"]), -math.log(1 / (1 + math.exp(-4.0))), rel_tol=1e-6)
    assert float(good["class_loss"]) == 0.0 and math.isclose(float(good["loss"]), float(good["rank_loss"]), rel_tol=1e-6)
    # pointwise labels do not influence the primary loss at all
    a = cost_loss(_t(0.3), _t(1.1), _t(-1.0), _t(1.0))
    b = cost_loss(_t(0.3), _t(1.1), _t(1.0), _t(-1.0))
    assert torch.equal(a["loss"], b["loss"])
    # shift invariance: adding a constant to both costs leaves the primary loss unchanged
    c = cost_loss(_t(0.3 + 7.0), _t(1.1 + 7.0))
    assert torch.allclose(a["loss"], c["loss"], atol=1e-6)


def test_cost_loss_pointwise_ablation_when_enabled():
    good = cost_loss(_t(-2.0), _t(2.0), _t(-1.0), _t(1.0), pointwise_classification=True)
    bad = cost_loss(_t(2.0), _t(-2.0), _t(-1.0), _t(1.0), pointwise_classification=True)
    assert float(good["rank_loss"]) < float(bad["rank_loss"]) and float(good["class_loss"]) < float(bad["class_loss"])
    assert math.isclose(float(good["loss"]), float(good["rank_loss"]) + float(good["class_loss"]), rel_tol=1e-6)
    assert math.isclose(float(cost_class_loss(_t(0.0), _t(1.0))), math.log(2), rel_tol=1e-6)  # threshold C = 0
    assert float(cost_class_loss(_t(-3.0), _t(-1.0))) < math.log(2) < float(cost_class_loss(_t(3.0), _t(-1.0)))
    assert float(cost_class_loss(_t(3.0), _t(1.0))) < math.log(2) < float(cost_class_loss(_t(-3.0), _t(1.0)))
    with pytest.raises(ValueError):
        cost_loss(_t(0.0), _t(1.0), pointwise_classification=True)


# ----------------------------------------------------------------------------- models

@pytest.fixture(scope="module")
def tiny_backbone_dir(tmp_path_factory, tokenizer, tiny_model):
    d = tmp_path_factory.mktemp("tiny_backbone")
    tiny_model.save_pretrained(d)
    tokenizer.save_pretrained(d)
    return d


def _batch(tokenizer, rows):
    from saferl.formatting import collate_dpo, encode_dpo_row

    return collate_dpo([encode_dpo_row(tokenizer, r) for r in rows], tokenizer.pad_token_id)


@pytest.mark.hf
def test_score_at_last_nonpadding_token_and_head_init(tokenizer, tiny_backbone_dir):
    from saferl.rm_models import ScoreModel, last_token_hidden

    m = ScoreModel.from_backbone(str(tiny_backbone_dir), ["reward", "cost"], "cpu")
    for h in m.heads.values():
        assert float(h.bias.abs().sum()) == 0.0 and h.weight.shape == (1, m.backbone.config.hidden_size)
    assert m.heads["reward"].weight.data_ptr() != m.heads["cost"].weight.data_ptr()  # distinct heads
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    out = last_token_hidden(hidden, mask)
    assert torch.equal(out[0], hidden[0, 2]) and torch.equal(out[1], hidden[1, 4])
    # the scored position is the appended EOS token
    rows = pairs_to_rm_rows([make_pair(0, r0="short", r1="a much longer answer " * 4)])
    b = _batch(tokenizer, rows[:1])
    idx = b["attention_mask"].sum(1) - 1
    assert all(int(b["input_ids"][i, idx[i]]) == tokenizer.eos_token_id for i in range(b["input_ids"].shape[0]))
    # padding does not change a sequence's score
    with torch.no_grad():
        s_batched = m(b["input_ids"], b["attention_mask"])["reward"]
        L = int(b["attention_mask"][0].sum())
        s_single = m(b["input_ids"][:1, :L], b["attention_mask"][:1, :L])["reward"]
    assert torch.allclose(s_batched[0], s_single[0], atol=1e-4)


@pytest.mark.hf
def test_shared_backbone_and_head_gradients(tokenizer, tiny_backbone_dir):
    from saferl.rm_models import ScoreModel
    from saferl.train_rm import microbatch_loss

    m = ScoreModel.from_backbone(str(tiny_backbone_dir), ["reward", "cost"], "cpu").train()
    # one backbone: both heads consume the same parameters
    assert len(list(m.backbone.parameters())) > 0 and sum(1 for _ in m.heads.parameters()) == 4
    pairs = [make_pair(i, better=0, safer=1, safe=(False, True)) for i in range(2)]
    h_rows = pairs_to_rm_rows(pairs, ["helpfulness"])
    s_rows = pairs_to_rm_rows(pairs, ["harmlessness"])
    spec = dict(ARCHITECTURES["shared"], cost_pointwise_classification=False)
    # helpfulness rows -> gradients on backbone + reward head only
    out = microbatch_loss(m, _batch(tokenizer, h_rows), h_rows, spec, "cpu", "fp32")
    out["loss"].sum().backward()
    assert m.heads["reward"].weight.grad is not None and m.heads["reward"].weight.grad.abs().sum() > 0
    assert m.heads["cost"].weight.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.backbone.parameters())
    m.zero_grad(set_to_none=True)
    # safety rows -> gradients on backbone + cost head only
    out = microbatch_loss(m, _batch(tokenizer, s_rows), s_rows, spec, "cpu", "fp32")
    out["loss"].sum().backward()
    assert m.heads["cost"].weight.grad is not None and m.heads["cost"].weight.grad.abs().sum() > 0
    assert m.heads["reward"].weight.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.backbone.parameters())


@pytest.mark.hf
def test_separate_models_share_no_parameters(tiny_backbone_dir):
    from saferl.rm_models import ScoreModel

    rm = ScoreModel.from_backbone(str(tiny_backbone_dir), ["reward"], "cpu")
    cm = ScoreModel.from_backbone(str(tiny_backbone_dir), ["cost"], "cpu")
    rm_ptrs = {p.data_ptr() for p in rm.parameters()}
    cm_ptrs = {p.data_ptr() for p in cm.parameters()}
    assert not (rm_ptrs & cm_ptrs)
    # identical initialization of the backbones (same checkpoint)
    for (n1, p1), (n2, p2) in zip(rm.backbone.named_parameters(), cm.backbone.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)


@pytest.mark.hf
def test_training_moves_scores_in_the_intended_direction(tokenizer, tiny_backbone_dir):
    """A few SGD steps on fixed rows must raise reward(chosen) - reward(rejected) and
    cost(harmful) - cost(safe), and push C(safe) < 0 < C(unsafe)."""
    from saferl.rm_models import ScoreModel
    from saferl.train_rm import microbatch_loss

    torch.manual_seed(0)
    m = ScoreModel.from_backbone(str(tiny_backbone_dir), ["reward", "cost"], "cpu").train()
    texts = [("Sure, here is a safe and helpful answer about gardening.", "You should hurt them; here is how to do it."),
             ("I cannot help with that; please seek support.", "Do this illegal thing step by step."),
             ("Here is a recipe for bread.", "Insulting rant about a group of people."),
             ("Let me explain photosynthesis.", "Instructions for making a weapon.")]
    pairs = [make_pair(i, better=0, safer=0, safe=(True, False), r0=a, r1=b) for i, (a, b) in enumerate(texts)]
    rows = pairs_to_rm_rows(pairs)
    b = _batch(tokenizer, rows)
    spec = dict(ARCHITECTURES["shared"], cost_pointwise_classification=False)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    def margins():
        with torch.no_grad():
            out = microbatch_loss(m, b, rows, spec, "cpu", "fp32")
        return out["margin"].clone(), out["loss"].sum().item()
    m0, l0 = margins()
    for _ in range(40):
        opt.zero_grad()
        microbatch_loss(m, b, rows, spec, "cpu", "fp32")["loss"].sum().backward()
        opt.step()
    m1, l1 = margins()
    assert l1 < l0
    assert float(m1.mean()) > float(m0.mean()) and float(m1.mean()) > 0  # reward: chosen above rejected; cost: harmful above safe (on average)
    with torch.no_grad():
        sc = m(b["input_ids"], b["attention_mask"])["cost"]
    B = b["batch_size"]
    s_idx = [j for j, r in enumerate(rows) if r["objective"] == "harmlessness"]
    c_safe = sc[torch.tensor(s_idx)]; c_harm = sc[torch.tensor([B + j for j in s_idx])]
    assert float(c_harm.mean()) > float(c_safe.mean())  # orientation: larger C = more harmful (safer_response_id ordering)


@pytest.mark.hf
def test_batched_inference_interface(tokenizer, tiny_backbone_dir, tmp_path):
    from saferl.rm_models import ScoreModel

    m = ScoreModel.from_backbone(str(tiny_backbone_dir), ["reward", "cost"], "cpu")
    prompts = ["How do I bake bread?", "Tell me a joke.", "What is 2+2?"]
    responses = ["Mix flour, water, yeast; bake.", "Why did the chicken cross the road?", "4"]
    r = m.score_reward(tokenizer, prompts, responses, batch_size=2, precision="fp32")
    c = m.score_cost(tokenizer, prompts, responses, batch_size=2, precision="fp32")
    assert len(r) == len(c) == 3 and all(isinstance(x, float) for x in r + c)
    r1 = m.score_reward(tokenizer, prompts, responses, batch_size=1, precision="fp32")
    assert all(math.isclose(a, b, abs_tol=1e-4) for a, b in zip(r, r1))
    with pytest.raises(KeyError):
        m.score_scalar(tokenizer, prompts, responses)
    m.save_pretrained(tmp_path / "ck", "float32")
    m2 = ScoreModel.load(str(tmp_path / "ck"), "cpu")
    r2 = m2.score_reward(tokenizer, prompts, responses, precision="fp32")
    assert all(math.isclose(a, b, abs_tol=1e-4) for a, b in zip(r, r2))


@pytest.mark.hf
def test_primary_loss_ignores_pointwise_labels_but_follows_safer_response_id(tokenizer, tiny_backbone_dir):
    from saferl.rm_models import ScoreModel
    from saferl.train_rm import arch_spec, microbatch_loss

    m = ScoreModel.from_backbone(str(tiny_backbone_dir), ["reward", "cost"], "cpu").eval()
    spec = arch_spec({"arch": "shared"})  # primary: flag absent -> False
    assert spec["cost_pointwise_classification"] is False
    base = make_pair(0, better=0, safer=1, safe=(False, True))
    flipped_labels = make_pair(0, better=0, safer=1, safe=(True, False))   # pointwise labels flipped only
    flipped_safer = make_pair(0, better=0, safer=0, safe=(False, True))    # safer_response_id flipped only
    def loss_of(pair):
        rows = pairs_to_rm_rows([pair], ["harmlessness"])
        with torch.no_grad():
            return microbatch_loss(m, _batch(tokenizer, rows), rows, spec, "cpu", "fp32")["loss"]
    assert torch.equal(loss_of(base), loss_of(flipped_labels))
    assert not torch.allclose(loss_of(base), loss_of(flipped_safer))
    # with the ablation flag the labels DO matter
    spec_abl = dict(spec, cost_pointwise_classification=True)
    rows = pairs_to_rm_rows([base], ["harmlessness"]); rows2 = pairs_to_rm_rows([flipped_labels], ["harmlessness"])
    with torch.no_grad():
        a = microbatch_loss(m, _batch(tokenizer, rows), rows, spec_abl, "cpu", "fp32")["loss"]
        b = microbatch_loss(m, _batch(tokenizer, rows2), rows2, spec_abl, "cpu", "fp32")["loss"]
    assert not torch.equal(a, b)


def test_all_architectures_use_identical_pairwise_supervision():
    pairs = synthetic_pool(100, "train", seed=13)
    rows = {a: {r["row_id"]: (r["prompt"], r["chosen"], r["rejected"]) for r in pairs_to_rm_rows(pairs, s["objectives"])}
            for a, s in ARCHITECTURES.items()}
    assert rows["scalar"] == rows["shared"]
    assert {k: v for k, v in rows["scalar"].items() if k.endswith(":H")} == rows["separate_reward"]
    assert {k: v for k, v in rows["scalar"].items() if k.endswith(":S")} == rows["separate_cost"]
    # the S row's chosen response is the safer_response_id one, for every architecture consuming it
    by_pair = {p["pair_id"]: p for p in pairs}
    for a in ("scalar", "shared", "separate_cost"):
        for rid, (_, chosen, _) in rows[a].items():
            if rid.endswith(":S"):
                p = by_pair[rid[:-2]]
                assert chosen == [p["response_0"], p["response_1"]][p["safer_response_id"]]


# ----------------------------------------------------------------------------- configs

def test_scientific_configs_share_common_hyperparameters_and_data():
    from saferl.train_rm import COMMON_HPARAM_KEYS

    cfgs = {a: load_config(str(ROOT / "configs" / f)) for a, f in ARCH_CONFIGS.items()}
    ref = cfgs["scalar"]
    for a, c in cfgs.items():
        assert c["arch"] == a
        for k in COMMON_HPARAM_KEYS:
            assert c.get(k) == ref.get(k), (a, k, c.get(k), ref.get(k))
        assert c["artifacts_dir"] == "artifacts" and c["seed"] == 42
    assert ref["model"] == "Qwen/Qwen2.5-1.5B-Instruct" and ref["full_finetune"] is True
    assert ref["max_epochs"] == 2 and ref["learning_rate"] == 2e-5 and ref["adam_beta2"] == 0.95
    assert ref["weight_decay"] == 0.1 and ref["warmup_ratio"] == 0.03 and ref["global_batch_size"] == 64
    assert ref["max_grad_norm"] == 1.0 and ref["precision"] == "bf16" and ref["gradient_checkpointing"] is True
    assert ref["max_prompt_length"] is None and ref["max_length"] is None  # from the DPO audit
    for a, c in cfgs.items():
        assert c["cost_pointwise_classification"] is False, a  # primary = pairwise BT only
    assert load_config(str(ROOT / "configs" / "rm_smoke.yaml"))["cost_pointwise_classification"] is False
    dpo = load_config(str(ROOT / "configs" / "dpo.yaml"))
    assert dpo["artifacts_dir"] == ref["artifacts_dir"] and dpo["model"] == ref["model"]


def test_seeds_do_not_change_data(tiny_backbone_dir, smoke_artifacts_dir):
    """Different training seeds load byte-identical rows; only the shuffle order changes."""
    from saferl.train_dpo import epoch_order

    rows42 = load_rm_rows(str(smoke_artifacts_dir), "train")
    rows43 = load_rm_rows(str(smoke_artifacts_dir), "train")
    assert rows42 == rows43
    assert epoch_order(len(rows42), 42, 0) != epoch_order(len(rows42), 43, 0)
    assert epoch_order(len(rows42), 42, 0) == epoch_order(len(rows42), 42, 0)


# ----------------------------------------------------------------------------- tiny end-to-end

@pytest.mark.hf
@pytest.mark.artifacts
def test_train_eval_aggregate_all_archs(tiny_backbone_dir, smoke_artifacts_dir, tmp_path):
    from saferl.aggregate_rm import aggregate
    from saferl.eval_rm import evaluate_checkpoint
    from saferl.train_rm import train

    for arch, fname in ARCH_CONFIGS.items():
        cfg = load_config(str(ROOT / "configs" / fname))
        cfg.update({"model": str(tiny_backbone_dir), "runs_dir": str(tmp_path / "runs"), "artifacts_dir": str(smoke_artifacts_dir),
                    "device": "cpu", "precision": "fp32", "gradient_checkpointing": False, "max_steps": 2, "eval_every_steps": 1,
                    "global_batch_size": 4, "per_device_batch_size": 2, "eval_batch_size": 4, "learning_rate": 1e-3,
                    "run_name": f"tiny_{arch}", "eval_splits": ["val"]})
        s = train(cfg)
        assert s["batch"]["gradient_accumulation_steps"] == 2 and s["total_steps"] == 2
        assert set(s["train_rows_per_objective"]) == set(ARCHITECTURES[arch]["objectives"])
        assert all(v == 32 for v in s["train_rows_per_objective"].values())
        assert s["parameters"]["heads"].keys() == set(ARCHITECTURES[arch]["heads"])
        assert s["best_step"] is not None and s["approx_train_flops"] > 0
        res = evaluate_checkpoint(cfg, Path(s["run_dir"]))
        g = res["splits"]["val"]["groups"]
        for o in ("helpfulness", "harmlessness"):
            assert (g[o]["n"] > 0) == (o in ARCHITECTURES[arch]["objectives"])
        if "harmlessness" in ARCHITECTURES[arch]["objectives"] and arch != "scalar":
            assert "class_acc" not in g["harmlessness"]  # primary: no pointwise term anywhere
            assert "cost_gap_unsafe_minus_safe" in g["harmlessness"] and g["harmlessness"]["n_responses"] == 2 * g["harmlessness"]["n"]
            assert math.isclose(g["harmlessness"]["loss"], g["harmlessness"]["rank_loss"], rel_tol=1e-9)
        if arch == "scalar":
            assert "class_acc" not in g["harmlessness"] and "cost_gap_unsafe_minus_safe" not in g["harmlessness"]
        assert res["cost_pointwise_classification"] is False
        agg = aggregate([Path(s["run_dir"])])
        assert agg["splits"]["val"]["combined_loss"]["n_seeds"] == 1
