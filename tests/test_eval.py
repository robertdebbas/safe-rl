"""Common evaluation pipeline: manifest, per-prompt seeds, shards, scorer layout, bootstrap, merge."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from saferl.eval_common import (EVALUATOR_TEMPLATE, build_eval_manifest, fixed_batches, load_eval_prompts, method_checkpoint,
                                prompt_seed, shard_batches, shard_prompt_indices, verify_checkpoint_rule)
from saferl.eval_merge import bootstrap_ci, merge_shards
from saferl.eval_score import PRIMARY_EVALUATORS, scoring_chunks
from saferl.utils import load_config, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- manifest / seeds / shards

@pytest.mark.artifacts
def test_manifest_is_deduplicated_deterministic_and_shared(artifacts_dir):
    info = build_eval_manifest(str(artifacts_dir))
    assert info["n_pairs"] == 2000 and info["n_prompts"] == 1960
    prompts = load_eval_prompts(str(artifacts_dir))
    assert len({p["prompt_hash"] for p in prompts}) == 1960
    assert [p["prompt_hash"] for p in prompts] == sorted(p["prompt_hash"] for p in prompts)   # deterministic order
    assert build_eval_manifest(str(artifacts_dir))["sha256"] == info["sha256"]              # byte-identical regeneration
    cfg = load_config(str(ROOT / "configs" / "eval.yaml"))
    # every method reads the same manifest (single artifacts_dir / file) and the same eval seed
    assert cfg["artifacts_dir"] == "artifacts" and len(cfg["methods"]) == 5
    for m in cfg["methods"].values():
        assert "{seed}" in m["checkpoint"] or m["seeds"] == [0]


def test_prompt_seed_is_deterministic_and_prompt_specific():
    assert prompt_seed(1, "abc") == prompt_seed(1, "abc")
    assert prompt_seed(1, "abc") != prompt_seed(2, "abc") and prompt_seed(1, "abc") != prompt_seed(1, "abd")
    assert 0 <= prompt_seed(20261030, "deadbeef") < 2 ** 63


def test_shards_are_disjoint_whole_batches_and_cover_everything():
    n, bs = 1960, 20
    for N in (1, 2, 3, 4, 7):
        parts = [shard_prompt_indices(n, bs, N, i) for i in range(N)]
        flat = [i for p in parts for i in p]
        assert sorted(flat) == list(range(n)) and len(flat) == len(set(flat))            # union = manifest, no duplicates
        for a in range(N):
            for b in range(a + 1, N):
                assert not (set(parts[a]) & set(parts[b]))
        # whole batches: every fixed batch lies in exactly one shard, so batch composition is layout-independent
        batches = fixed_batches(n, bs)
        for p in parts:
            assert all(all(i in set(p) for i in b) or not any(i in set(p) for i in b) for b in batches)
    assert [len(shard_prompt_indices(1960, 20, 4, i)) for i in range(4)] == [500, 500, 480, 480]
    with pytest.raises(ValueError):
        shard_batches(1960, 20, 4, 4)
    with pytest.raises(ValueError):
        shard_batches(40, 20, 3, 0)


def test_symmetric_final_checkpoint_rule_and_best_is_refused(tmp_path):
    cfg = load_config(str(ROOT / "configs" / "eval.yaml"))
    rules = {m: v["checkpoint_rule"] for m, v in cfg["methods"].items()}
    assert rules == {"base": "pretrained", "dpo": "dpo_final_2_epochs", "grpo_scalar": "grpo_final_1_prompt_epoch",
                     "grpo_shared": "grpo_final_1_prompt_epoch", "grpo_separate": "grpo_final_1_prompt_epoch"}
    for m, v in cfg["methods"].items():
        for seed in v["seeds"]:
            path = method_checkpoint(v, seed)
            assert "best" not in path, (m, path)
            assert m == "base" or path.endswith("/checkpoints/final"), (m, path)
    # guard: a 'best' path is refused for every rule
    for rule in ("dpo_final_2_epochs", "grpo_final_1_prompt_epoch", "pretrained"):
        with pytest.raises(ValueError, match="refuses the validation-selected"):
            verify_checkpoint_rule("runs/dpo_qwen2.5-1.5b_seed42/checkpoints/best", rule)
    # DPO: only the end-of-epoch-2 state passes
    d = tmp_path / "dpo_final"; d.mkdir()
    (d / "trainer_state.json").write_text(json.dumps({"step": 626, "epoch": 2.0, "is_epoch_end": True}))
    assert verify_checkpoint_rule(str(d), "dpo_final_2_epochs")["trainer_state"]["epoch"] == 2.0
    (d / "trainer_state.json").write_text(json.dumps({"step": 313, "epoch": 1.0, "is_epoch_end": True}))
    with pytest.raises(ValueError, match="end-of-epoch-2"):
        verify_checkpoint_rule(str(d), "dpo_final_2_epochs")
    (d / "trainer_state.json").write_text(json.dumps({"step": 500, "epoch": 1.6, "is_epoch_end": False, "is_best_at_save": True}))
    with pytest.raises(ValueError):
        verify_checkpoint_rule(str(d), "dpo_final_2_epochs")
    # GRPO: a partial prompt epoch is refused; the full epoch passes
    run = tmp_path / "grpo_run"; (run / "checkpoints" / "final").mkdir(parents=True)
    (run / "config_resolved.json").write_text(json.dumps({"n_train_prompts": 9054}))
    (run / "checkpoints" / "final" / "trainer_state.json").write_text(json.dumps({"prompt_epochs": 1, "n_prompts": 9000, "arm": "scalar"}))
    with pytest.raises(ValueError, match="expected the full epoch"):
        verify_checkpoint_rule(str(run / "checkpoints" / "final"), "grpo_final_1_prompt_epoch")
    (run / "checkpoints" / "final" / "trainer_state.json").write_text(json.dumps({"prompt_epochs": 1, "n_prompts": 9054, "arm": "scalar"}))
    assert verify_checkpoint_rule(str(run / "checkpoints" / "final"), "grpo_final_1_prompt_epoch")["trainer_state"]["n_prompts"] == 9054
    # pretrained: Hub id passes, a trained checkpoint does not
    assert verify_checkpoint_rule("Qwen/Qwen2.5-1.5B-Instruct", "pretrained") == {"rule": "pretrained"}
    with pytest.raises(ValueError):
        verify_checkpoint_rule(str(d), "pretrained")


def test_checkpoint_patterns_and_primary_evaluators():
    cfg = load_config(str(ROOT / "configs" / "eval.yaml"))
    assert method_checkpoint(cfg["methods"]["dpo"], 43) == "runs/dpo_qwen2.5-1.5b_seed43/checkpoints/final"
    assert method_checkpoint(cfg["methods"]["grpo_shared"], 44) == "runs/grpo_shared_constrained_qwen2.5-1.5b_seed44/checkpoints/final"
    assert method_checkpoint(cfg["methods"]["base"], 0) == "Qwen/Qwen2.5-1.5B-Instruct"
    ev = cfg["evaluators"]
    assert ev["reward"]["model"] == PRIMARY_EVALUATORS["reward"] == "PKU-Alignment/beaver-7b-unified-reward"
    assert ev["cost"]["model"] == PRIMARY_EVALUATORS["cost"] == "PKU-Alignment/beaver-7b-unified-cost"
    assert ev["template"] == EVALUATOR_TEMPLATE and ev["append_eos"] is True
    # no training RM/CM anywhere in the evaluation config or scoring module
    text = (ROOT / "configs" / "eval.yaml").read_text() + (ROOT / "src" / "saferl" / "eval_score.py").read_text()
    assert "rm_scalar" not in text and "rm_shared" not in text and "rm_separate" not in text and "ScoreModel.load" not in text
    assert cfg["bootstrap"]["n_replicates"] == 2000 and cfg["generation"] == {"max_prompt_length": 512, "max_new_tokens": 512,
                                                                              "temperature": 1.0, "top_p": 1.0, "batch_size": 20}


def test_scoring_chunks_never_cross_generation_batches_and_are_shard_invariant():
    n, gen_bs, eval_bs = 1960, 20, 16
    full = scoring_chunks(list(range(n)), eval_bs, gen_bs)
    assert all(len(c) <= eval_bs for c in full)
    assert all(len({i // gen_bs for i in c}) == 1 for c in full)                      # one generation batch per scoring chunk
    assert [len(c) for c in full[:4]] == [16, 4, 16, 4]
    # a shard's chunks (expressed in manifest indices) are exactly the subset of the unsharded chunks
    idx = shard_prompt_indices(n, gen_bs, 4, 2)
    sharded = [[idx[p] for p in c] for c in scoring_chunks(idx, eval_bs, gen_bs)]
    full_mi = [[i for i in c] for c in full]
    assert all(c in full_mi for c in sharded) and sum(len(c) for c in sharded) == len(idx)
    assert scoring_chunks([0, 1, 2], 16, None) == [[0, 1, 2]]


# ----------------------------------------------------------------------------- bootstrap

def test_bootstrap_equal_seed_weight_and_within_seed_resampling():
    rng = np.random.default_rng(0)
    per_seed = {42: rng.normal(0.0, 1.0, 1960), 43: rng.normal(1.0, 1.0, 1960), 44: rng.normal(5.0, 1.0, 100)}  # seed 44 small on purpose
    out = bootstrap_ci(per_seed, 2000, 7)
    assert out["n_replicates"] == 2000 and out["n_seeds"] == 3
    expected_point = np.mean([per_seed[s].mean() for s in per_seed])
    assert math.isclose(out["point"], expected_point, rel_tol=1e-12)              # equal weight per seed
    # With equal n per seed, flat pooling gives the same point estimate; the seed-stratified bootstrap differs by
    # respecting the policy-seed grouping in the resampling (the unequal-n case below only makes the distinction visible).
    pooled = np.concatenate(list(per_seed.values())).mean()
    assert abs(out["point"] - pooled) > 0.5
    assert out["ci_low"] < out["point"] < out["ci_high"]
    # within-seed resampling: with identical seeds the CI width shrinks like 1/sqrt(n_seeds) relative to one seed
    x = rng.normal(0, 1, 500)
    one = bootstrap_ci({1: x}, 2000, 3); three = bootstrap_ci({1: x, 2: x.copy(), 3: x.copy()}, 2000, 3)
    assert (three["ci_high"] - three["ci_low"]) < (one["ci_high"] - one["ci_low"]) * 0.8
    # deterministic given the bootstrap seed
    assert bootstrap_ci(per_seed, 2000, 7) == out


def test_bootstrap_never_pools_seeds():
    import inspect

    from saferl import eval_merge

    src = inspect.getsource(eval_merge.bootstrap_ci)
    assert "concatenate" not in src and "for s in seeds" in src


# ----------------------------------------------------------------------------- merge

def _write_shards(d: Path, hashes, layout, kind="scores", drop=None, dup=None):
    for i, idxs in enumerate(layout):
        rows = []
        for j in idxs:
            if drop is not None and j == drop:
                continue
            rows.append({"prompt_hash": hashes[j], "manifest_index": j, "reward_score": float(j), "cost_score": -float(j), "truncated": False,
                         "evaluator_input_truncated": False, "n_completion_tokens": 3, "method": "m", "policy_seed": 1})
            if dup is not None and j == dup:
                rows.append(dict(rows[-1]))
        write_jsonl(d / f"{kind}.shard{i:03d}of{len(layout):03d}.jsonl", rows)


def test_merge_verifies_coverage_and_matches_unsharded(tmp_path):
    hashes = [f"h{i:03d}" for i in range(10)]
    one = tmp_path / "one"; two = tmp_path / "two"
    _write_shards(one, hashes, [list(range(10))])
    _write_shards(two, hashes, [list(range(5)), list(range(5, 10))])
    a = merge_shards(one, "scores", hashes); b = merge_shards(two, "scores", hashes)
    assert a["rows"] == b["rows"] and a["n_shards"] == 1 and b["n_shards"] == 2
    bad = tmp_path / "missing"; _write_shards(bad, hashes, [list(range(5)), list(range(5, 10))], drop=7)
    with pytest.raises(ValueError, match="missing=1"):
        merge_shards(bad, "scores", hashes)
    bad2 = tmp_path / "dup"; _write_shards(bad2, hashes, [list(range(5)), list(range(5, 10))], dup=3)
    with pytest.raises(ValueError, match="duplicates=1"):
        merge_shards(bad2, "scores", hashes)
    bad3 = tmp_path / "incomplete"; write_jsonl(bad3 / "scores.shard000of002.jsonl", [])
    with pytest.raises(ValueError, match="missing shards"):
        merge_shards(bad3, "scores", hashes)


# ----------------------------------------------------------------------------- scorer layout / seeded sampling

@pytest.mark.hf
def test_pku_score_model_loader_and_last_token_score(tmp_path):
    import subprocess, sys

    from saferl.eval_score import PKUScoreModel, encode_for_evaluator, load_evaluator_tokenizer, score_pairs

    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tiny_evaluators.py"), "--out", str(tmp_path / "ev")], check=True, capture_output=True)
    m = PKUScoreModel.from_pretrained(str(tmp_path / "ev" / "reward"), "cpu", torch.float32)
    assert m.score_type == "reward" and m.score_head.bias is not None and not any(p.requires_grad for p in m.parameters())
    tok = load_evaluator_tokenizer(str(tmp_path / "ev" / "reward"))
    assert tok.pad_token_id == 32000
    enc = encode_for_evaluator(tok, ["hi"], ["hello there"], EVALUATOR_TEMPLATE, True, 512)
    ids = enc["input_ids"][0, : enc["lengths"][0]].tolist()
    assert ids[0] == tok.bos_token_id and ids[-1] == tok.eos_token_id                      # BOS added, </s> appended
    assert "BEGINNING OF CONVERSATION: USER: hi ASSISTANT:hello there" in tok.decode(ids, skip_special_tokens=True)
    s = score_pairs(m, tok, ["hi", "a much longer prompt about gardening"], ["hello there", "x"], "cpu", "fp32", EVALUATOR_TEMPLATE, True, 512, 2)
    s1 = score_pairs(m, tok, ["hi"], ["hello there"], "cpu", "fp32", EVALUATOR_TEMPLATE, True, 512, 1)
    assert math.isclose(s["scores"][0], s1["scores"][0], abs_tol=1e-4)                     # padding-invariant last-token score


@pytest.mark.hf
def test_per_prompt_seeded_sampling_is_batch_independent(tokenizer, tiny_model):
    from saferl.eval_generate import sample_with_prompt_seeds

    prompts = ["Say hi.", "Write a long sentence about the ocean please."]
    seeds = [prompt_seed(5, "a"), prompt_seed(5, "b")]
    both = sample_with_prompt_seeds(tiny_model, tokenizer, prompts, seeds, 64, 8, "cpu", "fp32")
    alone0 = sample_with_prompt_seeds(tiny_model, tokenizer, prompts[:1], seeds[:1], 64, 8, "cpu", "fp32")
    again = sample_with_prompt_seeds(tiny_model, tokenizer, prompts, seeds, 64, 8, "cpu", "fp32")
    assert both[0]["completion_ids"] == alone0[0]["completion_ids"]                         # same prompt, same seed -> same sample regardless of batch
    assert [r["completion_ids"] for r in both] == [r["completion_ids"] for r in again]        # reproducible
    other = sample_with_prompt_seeds(tiny_model, tokenizer, prompts[:1], [prompt_seed(6, "a")], 64, 8, "cpu", "fp32")
    assert other[0]["completion_ids"] != alone0[0]["completion_ids"] or len(alone0[0]["completion_ids"]) <= 1
    assert all(len(r["completion_ids"]) <= 8 for r in both)
