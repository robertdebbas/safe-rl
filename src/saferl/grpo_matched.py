"""Matched constrained-GRPO comparison for the REBEL study (PREPARED, not launched).

Same policy-training prompts, order, generation settings, frozen scorers, FIXED standardized reward/cost (REBEL calibration
mu/sigma), threshold d = -0.5 in standardized units and the same projected additive dual as the REBEL arms. Everything else is
the validated GRPO trainer, inherited unchanged: group-normalized advantages A = A_R - lambda A_C (invariant to the affine
standardization, so only the dual differs from the old runs), clipped surrogate (eps 0.2), KL to pi_0 (beta 0.02), minibatches of
whole groups. The remaining algorithm-specific differences vs REBEL are therefore exactly: (i) the policy update rule and
(ii) GRPO's explicit KL-to-pi_0 term. ``train_grpo`` itself is not modified; this module only subclasses it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import torch

from .grpo_data import prompt_order, rollout_batches
from .grpo_scoring import FrozenScorers
from .modeling import load_model, load_tokenizer
from .rebel_calibrate import ARM_HEADS
from .rebel_core import AdditiveDual, standardize
from .rebel_data import load_rebel_prompts
from .train_dpo import build_scheduler
from .train_grpo import GRPOTrainer, log, set_rng_states
from .utils import load_config, read_json, seed_everything, write_json


class StandardizedScorers:
    """Wraps FrozenScorers so every head returns (S - mu_S) / sigma_S with the FIXED calibration statistics."""

    def __init__(self, inner: FrozenScorers, stats: Dict[str, Any]):
        self.inner, self.stats = inner, stats
        self.arm, self.fingerprints, self.models = inner.arm, inner.fingerprints, inner.models

    @property
    def heads(self) -> List[str]:
        return self.inner.heads

    def score(self, tokenizer, prompts: Sequence[str], responses: Sequence[str]) -> Dict[str, torch.Tensor]:
        raw = self.inner.score(tokenizer, prompts, responses)
        return {h: standardize(raw[h], *self.stats[h]) for h in raw}

    def num_parameters(self) -> int:
        return self.inner.num_parameters()


class GRPOMatchedTrainer(GRPOTrainer):
    def setup(self) -> None:
        cfg = self.cfg
        seed_everything(self.seed)
        self.tok = load_tokenizer(cfg["policy_model"])
        self.pad = self.tok.pad_token_id
        self.train_prompts = load_rebel_prompts(cfg["artifacts_dir"], "train")
        self.monitor_prompts = load_rebel_prompts(cfg["artifacts_dir"], "monitoring")
        self.order = prompt_order(len(self.train_prompts), self.seed)
        self.batches = rollout_batches(self.order, self.B)
        self.total_opt_steps = sum(math.ceil(len(b) / self.groups_per_mb) for b in self.batches)
        if cfg.get("max_rollout_batches"):
            self.batches = self.batches[: int(cfg["max_rollout_batches"])]
        attn = cfg.get("attn_implementation")
        self.policy = load_model(cfg["policy_model"], self.device, torch.float32, bool(cfg["gradient_checkpointing"]), attn)
        self.ref = load_model(cfg["policy_model"], self.ref_device, torch.bfloat16 if self.precision == "bf16" else torch.float32, attn_implementation=attn)
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        rm_seed = cfg.get("scorer_rm_seed")
        if rm_seed is not None:
            for k, pth in cfg["scorer_checkpoints"].items():
                if f"_seed{int(rm_seed)}/" not in str(pth) and "tiny" not in str(pth) and "pytest" not in str(pth):
                    raise ValueError(f"scorer checkpoint {k}={pth} is not the RM seed-{rm_seed} checkpoint required by scorer_rm_seed")
        inner = FrozenScorers(self.arm, cfg["scorer_checkpoints"], self.scorer_device, self.precision, int(cfg.get("scorer_batch_size", 16)),
                              self.gen["max_prompt_length"], self.gen["max_prompt_length"] + self.gen["max_completion_length"])
        cal = read_json(cfg["calibration_file"])                       # the REBEL calibration of this arm (mu / sigma per head)
        if cal["arm"] != self.arm:
            raise ValueError(f"calibration file is for arm {cal['arm']!r}, run arm is {self.arm!r}")
        for k, fp in cal["scorer_checkpoints"].items():
            mine = inner.fingerprints.get(k, {})
            if fp.get("model.safetensors") != mine.get("model.safetensors") or fp.get("heads.pt") != mine.get("heads.pt"):
                raise ValueError(f"calibration was computed with a different {k} checkpoint than the one loaded")
        stats = {h: (float(cal["heads"][h]["mu"]), float(cal["heads"][h]["sigma"])) for h in ARM_HEADS[self.arm]}
        self.scorers = StandardizedScorers(inner, stats)
        self.dual = None
        self.calibration = {**cal, "units": "standardized", "d": float(cfg["d_std"]) if self.constrained else None}
        if self.constrained:
            if abs(float(cal["d_std"]) - float(cfg["d_std"])) > 1e-12:
                raise ValueError("config d_std differs from the calibration file's d_std")
            self.dual = AdditiveDual(d=float(cfg["d_std"]), beta=float(cfg["dual_step_size"]), lam=float(cfg.get("lambda_init", 1.0)),
                                     project=bool(cfg.get("dual_projection", True)))
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=float(cfg["learning_rate"]), betas=(float(cfg["adam_beta1"]), float(cfg["adam_beta2"])),
                                           eps=float(cfg["adam_epsilon"]), weight_decay=float(cfg["weight_decay"]))
        self.scheduler, self.warmup_steps = build_scheduler(self.optimizer, cfg, self.total_opt_steps)
        self.state = {"next_batch_index": 0, "opt_step": 0, "n_prompts": 0, "n_trajectories": 0, "n_generated_tokens": 0, "n_truncated": 0,
                      "n_zero_variance_groups": 0, "n_groups": 0, "elapsed_s": 0.0, "n_dual_updates": 0}
        self.monitor_history = []
        self.batch_log = open(self.run_dir / "rollout_batches.jsonl", "a", encoding="utf-8")
        write_json(self.run_dir / "config_resolved.json", {
            **{k: v for k, v in cfg.items() if not k.startswith("_")}, "algorithm": "grpo_matched", "device": self.device, "ref_device": self.ref_device,
            "scorer_device": self.scorer_device, "n_train_prompts": len(self.train_prompts), "n_rollout_batches": len(self.batches),
            "total_optimizer_steps": self.total_opt_steps, "warmup_steps": self.warmup_steps, "expected_trajectories": len(self.order) * self.G,
            "scorer_fingerprints": inner.fingerprints, "calibration": self.calibration, "n_gpus": self.n_gpus,
            "policy_parameters": sum(p.numel() for p in self.policy.parameters()), "scorer_parameters": inner.num_parameters(),
            "dual_rule": "additive projected: lambda <- max(0, lambda + beta (C_std_hat - d_std)); no EMA, no log parametrization",
            "scores": "standardized with the REBEL calibration (group-normalized advantages are invariant to this affine map)"})
        log.info("GRPO-matched arm=%s seed=%d prompts=%d batches=%d opt_steps=%d d_std=%s", self.arm, self.seed, len(self.train_prompts), len(self.batches),
                 self.total_opt_steps, self.dual.d if self.dual else None)

    def load_resume(self) -> bool:
        path = self.resume_path()
        if not path.exists():
            return False
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["seed"] != self.seed or ck["arm"] != self.arm or ck["order_check"] != [self.order[0], self.order[-1], len(self.order)]:
            raise RuntimeError("resume state does not match this run's seed / arm / prompt order")
        self.policy.load_state_dict(ck["policy"]); self.optimizer.load_state_dict(ck["optimizer"]); self.scheduler.load_state_dict(ck["scheduler"])
        self.state = ck["state"]
        if self.dual is not None:
            self.dual = AdditiveDual.from_state_dict(ck["dual"])
        self.monitor_history = ck["monitor_history"]
        set_rng_states(ck["rng"])
        log.info("resumed from batch %d (opt_step %d)", self.state["next_batch_index"], self.state["opt_step"])
        return True


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    summary = GRPOMatchedTrainer(cfg).train()
    if summary.get("stopped_for_diagnostic") and not summary.get("gates_passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
