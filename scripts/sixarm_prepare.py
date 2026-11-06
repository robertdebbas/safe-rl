"""Six-arm study (REBEL vs GRPO x scalar / separate-constrained d0 / d1): fixed prompt artifacts.

Builds artifacts/sixarm/ from the REBEL manifests (artifacts/rebel, built by saferl.rebel_data from UNUSED official train prompts):
  sixarm/grpo/{train,calibration,monitoring}_prompts.jsonl   read by the unmodified GRPO trainer (artifacts_dir/grpo)
  sixarm/rebel/{train,calibration,monitoring}_prompts.jsonl  read by the REBEL trainer (artifacts_dir/rebel)
  sixarm/prompt_exposure.json                                 the ORDERED 3,200 prompts of the first 200 rollout batches under
                                                              prompt_order(policy seed 42) -- identical for every arm -- plus hash
                                                              intersections with every RM/DPO split and the 1,960 evaluation prompts.
Read-only w.r.t. everything else. Usage: python scripts/sixarm_prepare.py [artifacts_dir] [--seed 42] [--batches 200] [--batch-size 16]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from saferl.grpo_data import prompt_order, rollout_batches  # noqa: E402
from saferl.rebel_data import load_rebel_prompts, prompt_key  # noqa: E402
from saferl.utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts", nargs="?", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hpval-prompts", type=int, default=640)
    a = ap.parse_args()
    art = Path(a.artifacts)
    src = art / "rebel"
    if not (src / "train_prompts.jsonl").exists():
        raise SystemExit("artifacts/rebel manifests missing: run scripts/run_rebel_prepare.sh first")
    out = art / "sixarm"
    for sub in ("grpo", "rebel"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for name in ("train", "calibration", "monitoring"):
            shutil.copyfile(src / f"{name}_prompts.jsonl", out / sub / f"{name}_prompts.jsonl")
    train = load_rebel_prompts(str(art), "train")
    order = prompt_order(len(train), a.seed)
    batches = rollout_batches(order, a.batch_size)[: a.batches]
    exposed = [train[i] for b in batches for i in b]
    hashes = [p["prompt_hash"] for p in exposed]
    assert len(set(hashes)) == len(hashes), "a prompt would be exposed twice"
    inter = {}
    for s in ("train", "val", "test_representative", "test_balanced"):
        keys = {prompt_key(p["prompt"]) for p in read_jsonl(art / "splits" / f"{s}.pairs.jsonl")}
        inter[f"split:{s}"] = len(set(hashes) & keys)
    ev = art / "eval" / "eval_prompts.jsonl"
    if ev.exists():
        inter["eval_prompts"] = len(set(hashes) & {p["prompt_hash"] for p in read_jsonl(ev)})
    for name in ("calibration", "monitoring"):
        inter[f"sixarm:{name}"] = len(set(hashes) & {p["prompt_hash"] for p in load_rebel_prompts(str(art), name)})
    if any(inter.values()):
        raise RuntimeError(f"prompt leakage: {inter}")
    info = {"policy_seed": a.seed, "n_batches": len(batches), "prompt_batch_size": a.batch_size, "n_prompts_exposed": len(hashes),
            "full_epoch_batches": len(rollout_batches(order, a.batch_size)), "n_train_prompts": len(train),
            "order_rule": "saferl.grpo_data.prompt_order(n_train_prompts, policy_seed) -- the same function in both trainers",
            "ordered_prompt_hashes": hashes, "hash_intersections": inter,
            "source": {"rebel_manifest_info": read_json(src / "manifest_info.json")["hashes"]},
            "hashes": {f"{sub}/{name}": sha256_file(out / sub / f"{name}_prompts.jsonl") for sub in ("grpo", "rebel") for name in ("train", "calibration", "monitoring")}}
    write_json(out / "prompt_exposure.json", info)
    # Pilot-training prompts (hyperparameter validation): 640 FRESH prompts from the still-unused official pool -- never in the main
    # manifest (so no resumption of a main run can reach them), never in calibration or monitoring. Written as a separate artifacts tree
    # so the pilots run the unmodified trainers. The held-out DEVELOPMENT set for selection is the 256 monitoring prompts, which no
    # run ever trains on.
    from collections import Counter
    from saferl.data import dataset_to_pairs, load_official
    from saferl.rebel_data import eligible_pool, source_quotas, stratified_take, used_prompt_keys
    from saferl.utils import load_config as _lc
    dcfg = _lc("configs/data.yaml", [])
    ds, _meta = load_official(dcfg["revision"], dcfg.get("config", "default"), dcfg.get("dataset", "PKU-Alignment/PKU-SafeRLHF"))
    used = used_prompt_keys(str(art))
    forbidden = set().union(*used.values()) | {p["prompt_hash"] for n in ("train", "calibration", "monitoring") for p in load_rebel_prompts(str(art), n)}
    pool, pool_report = eligible_pool(dataset_to_pairs(ds["train"], "train"), dataset_to_pairs(ds["test"], "test"), forbidden)
    ref = dict(Counter(p["prompt_source"] for p in train))
    import numpy as np
    dev = stratified_take(pool, source_quotas(ref, a.hpval_prompts), np.random.default_rng(a.seed + 7), set())
    dev_keys = {p["prompt_hash"] for p in dev}
    inter_dev = {"main_train": len(dev_keys & {p["prompt_hash"] for p in train}), "calibration": len(dev_keys & {p["prompt_hash"] for p in load_rebel_prompts(str(art), "calibration")}),
                 "monitoring": len(dev_keys & {p["prompt_hash"] for p in load_rebel_prompts(str(art), "monitoring")}), **{k: len(dev_keys & v) for k, v in used.items()}}
    if ev.exists():
        inter_dev["eval_prompts"] = len(dev_keys & {p["prompt_hash"] for p in read_jsonl(ev)})
    if any(inter_dev.values()):
        raise RuntimeError(f"pilot-training prompts leak: {inter_dev}")
    hp = art / "sixarm_hpval"
    for sub in ("grpo", "rebel"):
        (hp / sub).mkdir(parents=True, exist_ok=True)
        write_jsonl(hp / sub / "train_prompts.jsonl", dev)
        for name in ("calibration", "monitoring"):
            shutil.copyfile(src / f"{name}_prompts.jsonl", hp / sub / f"{name}_prompts.jsonl")
    write_json(hp / "hpval_info.json", {"pilot_training_prompts": len(dev), "source": "unused official train pool, stratified to the main mix, seed data_seed+7",
                                        "pool_after_exclusions": pool_report["eligible_prompts"], "by_source": dict(Counter(p["prompt_source"] for p in dev)),
                                        "prompt_hashes": [p["prompt_hash"] for p in dev], "hash_intersections": inter_dev,
                                        "development_set_for_selection": "the 256 monitoring prompts (monitoring_prompts.jsonl): never trained on by any pilot or main run",
                                        "hashes": {f"{sub}/train": sha256_file(hp / sub / "train_prompts.jsonl") for sub in ("grpo", "rebel")}})
    print({k: v for k, v in info.items() if k not in ("ordered_prompt_hashes", "source", "hashes")})


if __name__ == "__main__":
    main()
