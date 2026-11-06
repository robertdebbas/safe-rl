"""End-to-end trainer/evaluator/aggregator on a tiny random Qwen2 model + the smoke artifacts.
Validates: data loading, tokenization, reference log-prob caching, forward/backward, DPO loss,
checkpoint save/load/selection, evaluation, aggregation. Numbers are meaningless."""
import pytest
import yaml

from saferl.utils import read_json

pytestmark = [pytest.mark.hf, pytest.mark.artifacts]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory, tokenizer, tiny_model):
    d = tmp_path_factory.mktemp("tiny_model")
    tiny_model.save_pretrained(d)
    tokenizer.save_pretrained(d)
    return d


def _cfg(root, tiny_model_dir, tmp_path, seed):
    cfg = yaml.safe_load(open(root / "configs" / "dpo_smoke.yaml"))
    cfg.update({
        "model": str(tiny_model_dir), "runs_dir": str(tmp_path / "runs"), "artifacts_dir": str(root / "artifacts" / "smoke"),
        "ref_cache_dir": str(tmp_path / "refcache"), "device": "cpu", "precision": "fp32", "gradient_checkpointing": False,
        "max_steps": 3, "eval_every_steps": 2, "global_batch_size": 4, "per_device_batch_size": 2, "eval_batch_size": 4,
        "learning_rate": 1e-3, "seed": seed, "run_name": "tiny",
    })
    return cfg


def test_train_eval_aggregate_roundtrip(tiny_model_dir, smoke_artifacts_dir, tmp_path):
    from pathlib import Path

    from saferl.aggregate import aggregate
    from saferl.eval_dpo import evaluate_checkpoint
    from saferl.train_dpo import train

    root = Path(__file__).resolve().parents[1]
    summaries = []
    for seed in (1, 2):
        cfg = _cfg(root, tiny_model_dir, tmp_path, seed)
        if seed == 2:  # second seed: cheaper (1 step, val only) - checks cache reuse and seed independence
            cfg.update({"max_steps": 1, "eval_splits": ["val"]})
        s = train(cfg)
        summaries.append(s)
        run_dir = Path(s["run_dir"])
        # global batch exactly constructed
        assert s["batch"] == {"world_size": 1, "per_device_batch_size": 2, "gradient_accumulation_steps": 2, "global_batch_size": 4}
        assert s["total_steps"] == cfg["max_steps"] and s["train_rows_helpfulness"] == s["train_rows_harmlessness"] == 32
        # step-0 validation: policy == reference -> loss log 2, margin 0
        v0 = s["val_curve"][0]
        assert abs(v0["combined_loss"] - 0.6931) < 1e-3
        # eval points: step 0, 2, and final step 3
        assert [v["step"] for v in s["val_curve"]] == ([0, 2, 3] if seed == 1 else [0, 1])
        assert s["best_step"] is not None and (run_dir / "checkpoints" / "best").exists()
        assert (run_dir / "checkpoints" / "final").exists() and s["final_step"] == s["total_steps"]
        assert (run_dir / "checkpoints" / "final").resolve() == (run_dir / "checkpoints" / f"step_{s['total_steps']:06d}").resolve()
        from saferl.eval_common import verify_checkpoint_rule
        with pytest.raises(ValueError):  # this tiny run stopped by max_steps (< 2 epochs): not a valid PRIMARY DPO checkpoint
            verify_checkpoint_rule(str(run_dir / "checkpoints" / "final"), "dpo_final_2_epochs")
        assert (run_dir / "checkpoints" / f"step_{s['best_step']:06d}" / "model.safetensors").exists()
        assert (run_dir / "metrics.jsonl").exists() and (run_dir / "val_history.json").exists()
        # training with lr=1e-3 on a tiny model must change the loss away from log 2 by the end
        assert s["val_curve"][-1]["combined_loss"] != pytest.approx(0.6931, abs=1e-4)
        res = evaluate_checkpoint(cfg, run_dir)
        for split in cfg["eval_splits"]:
            m = res["splits"][split]
            assert m["groups"]["overall"]["n"] == 16
            assert m["groups"]["helpfulness"]["n"] == m["groups"]["harmlessness"]["n"] == 8
            d = m["conflict_diagnostic"]
            assert d["favors_helpfulness_preferred"]["count"] + d["favors_safety_preferred"]["count"] + d["ties"]["count"] == d["n_conflict_pairs"]
            assert (run_dir / "eval" / f"{split}.json").exists()
    # second seed reused the cached reference log-probs
    info = read_json(Path(summaries[1]["run_dir"]) / "ref_logprobs_info.json")
    assert info["train"]["from_cache"] and info["val"]["from_cache"]
    # seeds differ in trajectory (different shuffles) but share data
    assert summaries[0]["val_curve"][-1]["combined_loss"] != summaries[1]["val_curve"][-1]["combined_loss"]
    agg = aggregate([Path(s["run_dir"]) for s in summaries])
    assert agg["n_seeds"] == 2
    assert agg["splits"]["val"]["groups"]["overall"]["loss"]["n_seeds"] == 2
    assert agg["splits"]["val"]["conflict_diagnostic"]["n_conflict_pairs"] == 4
    assert set(agg["splits"]) == {"val"}  # intersection of evaluated splits


def test_global_batch_must_be_exact(tiny_model_dir, smoke_artifacts_dir, tmp_path):
    from pathlib import Path

    from saferl.train_dpo import resolve_batch_config

    root = Path(__file__).resolve().parents[1]
    cfg = _cfg(root, tiny_model_dir, tmp_path, 1)
    cfg["global_batch_size"], cfg["per_device_batch_size"] = 64, 6
    with pytest.raises(RuntimeError):
        resolve_batch_config(cfg)
    cfg["per_device_batch_size"] = 4
    assert resolve_batch_config(cfg)["gradient_accumulation_steps"] == 16
