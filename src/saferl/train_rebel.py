"""(Constrained) REBEL policy training -- an isolated module; the GRPO trainer is untouched.

Arms:  scalar   -> REBEL on the frozen scalar RM F (standardized F~), no constraint
       shared   -> constrained REBEL, R~ and C~ from the frozen shared two-head model
       separate -> constrained REBEL, R~ and C~ from two frozen independent models

One prompt epoch over the fixed REBEL training-prompt manifest (same seed-dependent order as GRPO). Per rollout batch
(B prompts x G completions sampled from the current weights pi_t, full-distribution sampling, top_k pinned to 0):
  1. generate all completions under pi_t;              2. store their sequence log-probs under pi_t (no grad, never recomputed);
  3. score with the frozen scorer(s), apply the FIXED calibration (mu, sigma);  4. fixed random disjoint pairs + targets
     z = dR~ - lambda_t dC~ (or dF~) with the SAME lambda_t for the whole batch;   5. minibatches of pairs, one optimizer step each,
     loss = mean over pairs of ((dl_theta(y1) - dl_theta(y2)) / eta - z)^2, inner_epochs passes (default 1: every pair used once);
  6. ONE dual update lambda <- max(0, lambda + beta (C_hat - d)) with C_hat = mean standardized cost of ALL rollouts of pi_t
     (pre-update policy: the cost estimate lags the updated policy by one iteration);   7. next batch with the updated policy.
Monitoring: KL(pi_t || pi_0) on the batch's own rollouts (k3 estimator); KL(pi_t || pi_{t+1}) via one extra no-grad pass after the
primal steps; regression fit before/after the update; length / truncation / EOS; reward-length correlation; therapy-list style density.
Optional, OFF by default and explicit: ``kl_to_initial_coef`` > 0 shapes the per-response reward as in the REBEL paper's Eq. 22,
r <- r - gamma (log pi_t(y|x) - log pi_0(y|x)); it is never mixed into the proximal term.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .grpo_core import DualState
from .grpo_data import prompt_order, rollout_batches
from .grpo_diag import ascii_stats, collapse_check, evaluate_gates, records_finite, tail_mass_and_entropy
from .grpo_rollout import batched_token_logprobs, collate_trajectories, generate_completions, token_logprobs
from .grpo_scoring import FrozenScorers
from .modeling import load_model, load_tokenizer, save_checkpoint
from .rebel_calibrate import ARM_HEADS
from .rebel_core import AdditiveDual, kl_shaping, kl_stats, pair_indices, rebel_loss, standardize
from .rebel_data import DUMP_GEN_SEED, MONITOR_GEN_SEEDS, load_rebel_prompts
from .train_dpo import build_scheduler, free_cache
from .utils import get_logger, load_config, pick_device, read_json, seed_everything, world_size, write_json, write_jsonl

log = get_logger("saferl.train_rebel")

COMMON_POLICY_KEYS = (
    "policy_model", "learning_rate", "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay", "lr_scheduler", "warmup_ratio", "max_grad_norm",
    "precision", "gradient_checkpointing", "prompt_epochs", "prompt_batch_size", "group_size", "pairs_per_minibatch", "inner_epochs", "rebel_eta",
    "generation", "lambda_init", "dual_step_size", "d_std", "dual_projection", "kl_to_initial_coef", "eval_every_batches", "resume_every_batches",
    "checkpoint_every_batches", "dump_every_batches", "data_seed", "score_units", "dual_rule", "dual_learning_rate", "cost_ema_decay",
)
CONSTRAINED_ARMS = ("shared", "separate")
THERAPY_RE = re.compile(r"\b(therap\w*|counsel\w*|mindfulness|meditation|trauma\w*|mental health|support group\w*|coping|self-care|anxiety|"
                        r"stress reduction|emotional\w*|psycholog\w*|CBT|EMDR|journaling|well-being|wellbeing)\b", re.I)
REFUSAL_RE = re.compile(r"\b(I can(?:not|'t)|I'm (?:unable|not able)|I (?:won't|will not)|not (?:appropriate|able to)|cannot (?:provide|assist|help)|"
                        r"I'm sorry|Sorry)\b", re.I)


def is_constrained(arm: str) -> bool:
    return arm in CONSTRAINED_ARMS


def gen_seed(seed: int, batch_index: int) -> int:
    """Rollout sampling seed = f(policy seed, batch index): resume-safe, identical across arms."""
    return int(seed) * 1_000_003 + int(batch_index)


def rng_states() -> Dict[str, Any]:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_states(st: Dict[str, Any]) -> None:
    random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch_cpu"])
    if "torch_cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["torch_cuda"])


def device_memory(device: str) -> Dict[str, float]:
    if device.startswith("cuda"):
        return {"allocated_gb": torch.cuda.memory_allocated() / 2**30, "max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_gb": torch.cuda.memory_reserved() / 2**30}
    if device == "mps":
        return {"allocated_gb": torch.mps.current_allocated_memory() / 2**30}
    return {}


def style_features(texts: List[str]) -> Dict[str, float]:
    """Reward-hacking diagnostics: therapy/support-resource term density, refusal openings, run-on sentences."""
    th, ref, longest = [], [], []
    for t in texts:
        w = max(1, len(t.split()))
        th.append(100.0 * len(THERAPY_RE.findall(t)) / w)
        ref.append(bool(REFUSAL_RE.search(t[:300])))
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
        longest.append(max((len(s.split()) for s in sents), default=0))
    return {"therapy_terms_per_100w": float(np.mean(th)) if th else 0.0, "frac_any_therapy_term": float(np.mean([x > 0 for x in th])) if th else 0.0,
            "refusal_open_rate": float(np.mean(ref)) if ref else 0.0, "longest_sentence_words_mean": float(np.mean(longest)) if longest else 0.0}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


class HardFailure(RuntimeError):
    """NaN/Inf, invalid log-probabilities, broken dual state or checkpoint corruption -> exit 3."""


class REBELTrainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.arm = cfg["arm"]
        if self.arm not in ARM_HEADS:
            raise ValueError(f"unknown arm {self.arm!r}")
        self.constrained = is_constrained(self.arm)
        self.seed = int(cfg["seed"])
        if world_size() != 1:
            raise RuntimeError("single-process trainer")
        if int(cfg.get("prompt_epochs", 1)) != 1:
            raise ValueError("exactly ONE prompt epoch is pre-specified for all arms")
        self.device = pick_device(cfg.get("device", "auto"))
        self.ref_device = cfg.get("ref_device") or self.device
        self.scorer_device = cfg.get("scorer_device") or self.device
        self.precision = cfg["precision"]
        self.gen_precision = cfg.get("generation_precision") or self.precision
        self.run_dir = Path(cfg["runs_dir"]) / f"{cfg['run_name']}_seed{self.seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.gen = cfg["generation"]
        self.B, self.G = int(cfg["prompt_batch_size"]), int(cfg["group_size"])
        if self.G % 2:
            raise ValueError("group_size must be even (disjoint pairs)")
        self.pairs_per_mb = int(cfg["pairs_per_minibatch"])
        self.inner_epochs = int(cfg.get("inner_epochs", 1))
        self.micro = int(cfg.get("policy_micro_batch_size", 8))          # trajectories per forward/backward (even -> whole pairs)
        if self.micro % 2:
            raise ValueError("policy_micro_batch_size must be even (pairs are never split across micro-batches)")
        self.eta = float(cfg["rebel_eta"])
        if self.eta <= 0:
            raise ValueError("rebel_eta must be positive")
        self.kl_coef = float(cfg.get("kl_to_initial_coef", 0.0) or 0.0)
        # Units of the reference-KL reward shaping (REBEL paper Eq. 22): "sequence_sum" = gamma * (l_t(y) - l_0(y)) summed over tokens
        # (the paper's form; at gamma = 0.02 and the KL levels GRPO reached this is 1-4 raw units, as large as dR itself);
        # "token_mean" = gamma * mean_t (log pi_t - log pi_0), the per-token weighting GRPO's beta applies (0.01 at the same KL).
        self.kl_units = str(cfg.get("kl_shaping_units", "token_mean"))
        if self.kl_units not in ("sequence_sum", "token_mean"):
            raise ValueError(f"unknown kl_shaping_units {self.kl_units!r}")
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        cfg = self.cfg
        seed_everything(self.seed)
        self.tok = load_tokenizer(cfg["policy_model"])
        self.pad = self.tok.pad_token_id
        self.train_prompts = load_rebel_prompts(cfg["artifacts_dir"], "train")
        self.monitor_prompts = load_rebel_prompts(cfg["artifacts_dir"], "monitoring")
        self.order = prompt_order(len(self.train_prompts), self.seed)        # identical permutation rule to GRPO
        self.batches = rollout_batches(self.order, self.B)
        self.total_opt_steps = sum(math.ceil(len(b) * (self.G // 2) / self.pairs_per_mb) for b in self.batches) * self.inner_epochs
        if cfg.get("schedule_total_batches"):   # pilots: define the LR schedule over the MAIN run's batch count (e.g. 566) on a smaller manifest
            self.total_opt_steps = int(cfg["schedule_total_batches"]) * math.ceil(self.B * (self.G // 2) / self.pairs_per_mb) * self.inner_epochs
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
        self.scorers = FrozenScorers(self.arm, cfg["scorer_checkpoints"], self.scorer_device, self.precision, int(cfg.get("scorer_batch_size", 16)),
                                     self.gen["max_prompt_length"], self.gen["max_prompt_length"] + self.gen["max_completion_length"])
        for m in self.scorers.models.values():
            for p in m.parameters():
                if p.requires_grad:
                    raise RuntimeError("scorer parameters must be frozen")
        # fixed calibration (mu, sigma per head) computed on the common pi_0 calibration responses with THESE scorer checkpoints
        if cfg.get("calibration_file"):
            self.calibration = read_json(cfg["calibration_file"])
            if self.calibration["arm"] != self.arm:
                raise ValueError(f"calibration file is for arm {self.calibration['arm']!r}, run arm is {self.arm!r}")
            for k, fp in self.calibration["scorer_checkpoints"].items():
                mine = self.scorers.fingerprints.get(k, {})
                if fp.get("model.safetensors") != mine.get("model.safetensors") or fp.get("heads.pt") != mine.get("heads.pt"):
                    raise ValueError(f"calibration was computed with a different {k} checkpoint than the one loaded")
        elif self.constrained or str(cfg.get("score_units", "raw")) != "raw":
            raise ValueError("a calibration file is required for constrained arms and for standardized scores")
        else:  # unconstrained arm on raw scores (e.g. scalar REBEL): nothing to calibrate
            self.calibration = {"arm": self.arm, "scorer_checkpoints": dict(self.scorers.fingerprints), "generations_sha256": None, "d": None}
        # Score units. "raw": the scorer outputs as they are (six-arm study; the calibration file carries d in raw cost units,
        # exactly as the GRPO arms use it). "standardized": S~ = (S - mu)/sigma from the calibration's per-head statistics.
        self.units = str(cfg.get("score_units", "raw"))
        if self.units == "standardized":
            self.stats = {h: (float(self.calibration["heads"][h]["mu"]), float(self.calibration["heads"][h]["sigma"])) for h in ARM_HEADS[self.arm]}
        elif self.units == "raw":
            self.stats = {h: (0.0, 1.0) for h in ARM_HEADS[self.arm]}
        else:
            raise ValueError(f"unknown score_units {self.units!r}")
        for h, (mu, sg) in self.stats.items():
            if not (np.isfinite(mu) and np.isfinite(sg) and sg > 0):
                raise ValueError(f"degenerate calibration for head {h}: mu={mu} sigma={sg}")
        # Dual rule. "grpo_log_lambda": GRPO's validated convention (mu <- mu + eta * lambda * (C_ema - d), lambda = exp(mu),
        # EMA of the batch cost, once per rollout batch after the primal steps) with the SAME hyperparameters as the GRPO arms.
        # "additive_projected": lambda <- max(0, lambda + beta (C_hat - d)) in standardized units (earlier proposal; kept as an option).
        self.dual_rule = str(cfg.get("dual_rule", "grpo_log_lambda"))
        self.dual = None
        if self.constrained:
            if self.dual_rule == "grpo_log_lambda":
                d = float(self.calibration["d"])
                if self.units != "raw":
                    raise ValueError("grpo_log_lambda uses the calibration's raw-unit d; set score_units: raw")
                self.dual = DualState(d=d, eta=float(cfg["dual_learning_rate"]), log_lambda=math.log(float(cfg.get("lambda_init", 1.0))),
                                      ema_decay=float(cfg.get("cost_ema_decay", 0.9)))
            elif self.dual_rule == "additive_projected":
                d_std = float(cfg["d_std"])
                if self.units != "standardized" or abs(float(self.calibration["d_std"]) - d_std) > 1e-12:
                    raise ValueError("additive_projected needs score_units: standardized and config d_std equal to the calibration's d_std")
                self.dual = AdditiveDual(d=d_std, beta=float(cfg["dual_step_size"]), lam=float(cfg.get("lambda_init", 1.0)), project=bool(cfg.get("dual_projection", True)))
            else:
                raise ValueError(f"unknown dual_rule {self.dual_rule!r}")
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=float(cfg["learning_rate"]), betas=(float(cfg["adam_beta1"]), float(cfg["adam_beta2"])),
                                           eps=float(cfg["adam_epsilon"]), weight_decay=float(cfg["weight_decay"]))
        self.scheduler, self.warmup_steps = build_scheduler(self.optimizer, cfg, self.total_opt_steps)
        self.state = {"next_batch_index": 0, "opt_step": 0, "n_prompts": 0, "n_trajectories": 0, "n_pairs": 0, "n_generated_tokens": 0, "n_truncated": 0,
                      "n_eos": 0, "elapsed_s": 0.0, "n_dual_updates": 0, "n_batches_violating": 0}
        self.monitor_history: List[Dict[str, Any]] = []
        self.batch_log = open(self.run_dir / "rollout_batches.jsonl", "a", encoding="utf-8")
        write_json(self.run_dir / "config_resolved.json", {
            **{k: v for k, v in cfg.items() if not k.startswith("_")}, "algorithm": "rebel", "device": self.device, "ref_device": self.ref_device,
            "scorer_device": self.scorer_device, "n_train_prompts": len(self.train_prompts), "n_rollout_batches": len(self.batches),
            "total_optimizer_steps": self.total_opt_steps, "warmup_steps": self.warmup_steps, "expected_trajectories": len(self.order) * self.G,
            "expected_pairs": len(self.order) * (self.G // 2), "scorer_fingerprints": self.scorers.fingerprints, "calibration": self.calibration,
            "n_gpus": self.n_gpus, "policy_parameters": sum(p.numel() for p in self.policy.parameters()), "scorer_parameters": self.scorers.num_parameters(),
            "score_units": self.units, "dual_rule": self.dual_rule, "d": (self.dual.d if self.dual else None),
            "old_policy_logprobs": "stored once per rollout batch before any update (never recomputed with the updated policy)"})
        log.info("REBEL arm=%s seed=%d prompts=%d batches=%d opt_steps=%d eta=%s d_std=%s device=%s", self.arm, self.seed, len(self.train_prompts),
                 len(self.batches), self.total_opt_steps, self.eta, self.dual.d if self.dual else None, self.device)

    # ------------------------------------------------------------------ scoring with the fixed calibration
    def score(self, prompts: List[str], texts: List[str]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raw = self.scorers.score(self.tok, prompts, texts)
        std = {h: standardize(raw[h], *self.stats[h]) for h in self.stats}
        return raw, std

    # ------------------------------------------------------------------ resume
    def resume_path(self) -> Path:
        return self.run_dir / "resume" / "latest.pt"

    def save_resume(self) -> None:
        path = self.resume_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save({"policy": {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}, "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(), "state": self.state, "dual": self.dual.state_dict() if self.dual else None, "rng": rng_states(),
                    "monitor_history": self.monitor_history, "seed": self.seed, "arm": self.arm, "algorithm": "rebel",
                    "calibration_sha": self.calibration.get("generations_sha256"), "order_check": [self.order[0], self.order[-1], len(self.order)]}, tmp)
        os.replace(tmp, path)
        try:  # corruption check: the file must load and carry the progress counters
            ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            if ck["state"]["next_batch_index"] != self.state["next_batch_index"]:
                raise HardFailure("resume file does not reflect the saved progress")
            del ck
        except HardFailure:
            raise
        except Exception as e:  # pragma: no cover
            raise HardFailure(f"resume checkpoint unreadable: {e}")
        log.info("resume state saved at batch %d (%s)", self.state["next_batch_index"], path)

    def load_resume(self) -> bool:
        path = self.resume_path()
        if not path.exists():
            return False
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["seed"] != self.seed or ck["arm"] != self.arm or ck.get("algorithm") != "rebel" or ck["order_check"] != [self.order[0], self.order[-1], len(self.order)]:
            raise RuntimeError("resume state does not match this run's seed / arm / algorithm / prompt order")
        if ck.get("calibration_sha") != self.calibration.get("generations_sha256"):
            raise RuntimeError("resume state was produced with a different calibration set")
        self.policy.load_state_dict(ck["policy"]); self.optimizer.load_state_dict(ck["optimizer"]); self.scheduler.load_state_dict(ck["scheduler"])
        self.state = ck["state"]
        if self.dual is not None:
            if self.dual_rule == "grpo_log_lambda":
                self.dual = DualState.from_state_dict(ck["dual"])
                self.dual.eta = float(self.cfg["dual_learning_rate"])
            else:
                self.dual = AdditiveDual.from_state_dict(ck["dual"])
        self.monitor_history = ck["monitor_history"]
        set_rng_states(ck["rng"])
        log.info("resumed from batch %d (opt_step %d)", self.state["next_batch_index"], self.state["opt_step"])
        return True

    # ------------------------------------------------------------------ one rollout batch
    def primal_steps(self, trajs: List[Dict[str, Any]], old_sum: torch.Tensor, pairs: List[Tuple[int, int]], target: torch.Tensor) -> Dict[str, Any]:
        """Minibatches of whole pairs; one optimizer step per minibatch; ``old_sum`` and ``target`` are constants."""
        old_sum = old_sum.detach().clone(); target = target.detach().clone()
        agg = {"loss": 0.0, "n_pairs": 0, "grad_norm": [], "steps": 0, "first_pred": [], "first_target": []}
        self.policy.train()
        n_pairs = len(pairs)
        micro_pairs = max(1, self.micro // 2)
        for _ in range(self.inner_epochs):
            for mstart in range(0, n_pairs, self.pairs_per_mb):
                mb = list(range(mstart, min(mstart + self.pairs_per_mb, n_pairs)))
                n_mb = len(mb)
                self.optimizer.zero_grad(set_to_none=True)
                for ustart in range(0, n_mb, micro_pairs):
                    upairs = [pairs[i] for i in mb[ustart:ustart + micro_pairs]]
                    tidx = [i for pr in upairs for i in pr]
                    b = collate_trajectories([trajs[i] for i in tidx], self.pad)
                    r = token_logprobs(self.policy, b, self.device, self.precision)
                    lp = r["sum_logprob"].float()                               # SUM over completion tokens (EOS included), prompt/pad masked
                    lp1, lp2 = lp[0::2], lp[1::2]
                    o1 = torch.tensor([float(old_sum[a]) for a, _ in upairs], device=lp.device); o2 = torch.tensor([float(old_sum[c]) for _, c in upairs], device=lp.device)
                    z = torch.tensor([float(target[i]) for i in mb[ustart:ustart + micro_pairs]], device=lp.device)
                    out = rebel_loss(lp1, lp2, o1, o2, z, self.eta)
                    (out["loss"] * len(upairs) / n_mb).backward()               # mean over the minibatch's pairs (chunked accumulation)
                    agg["loss"] += out["loss_value"] * len(upairs); agg["n_pairs"] += len(upairs)
                    if not agg["first_pred"]:
                        agg["first_pred"] = out["pred"][:2].tolist(); agg["first_target"] = z[:2].tolist()
                gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.cfg["max_grad_norm"]))
                if not math.isfinite(float(gn)):
                    raise HardFailure(f"non-finite gradient norm {gn}")
                self.optimizer.step(); self.scheduler.step()
                self.state["opt_step"] += 1; agg["steps"] += 1; agg["grad_norm"].append(float(gn))
        self.policy.eval()
        return agg

    def rollout_batch(self, batch_index: int, prompt_indices: List[int]) -> Dict[str, Any]:
        t0 = time.time()
        prompts = [self.train_prompts[i]["prompt"] for i in prompt_indices]
        lam_t = self.dual.lam if self.dual else None
        trajs = generate_completions(self.policy, self.tok, prompts, self.G, self.gen["max_prompt_length"], self.gen["max_completion_length"], self.device,
                                     self.gen_precision, self.gen["temperature"], self.gen["top_p"], self.gen["repetition_penalty"], seed=gen_seed(self.seed, batch_index))
        t_gen = time.time() - t0
        texts = [t["text"] for t in trajs]
        raw, std = self.score([prompts[t["prompt_index"]] for t in trajs], texts)
        old_tok = batched_token_logprobs(self.policy, trajs, self.pad, self.micro, self.device, self.precision)   # pi_t, stored, never recomputed
        ref_tok = batched_token_logprobs(self.ref, trajs, self.pad, self.micro, self.ref_device, self.precision)
        old_sum = torch.stack([x.sum() for x in old_tok]); ref_sum = torch.stack([x.sum() for x in ref_tok])
        if not torch.isfinite(old_sum).all() or bool((torch.cat(old_tok) > 1e-4).any()):
            raise HardFailure("invalid old-policy log-probabilities (non-finite or > 0)")
        t_lp = time.time() - t0 - t_gen
        if self.constrained:
            R, C = std["reward"], std["cost"]
            base = R - float(lam_t) * C                                         # per-response Lagrangian score (standardized units)
        else:
            base = std["scalar"]
        shaping = torch.zeros_like(base)
        if self.kl_coef > 0:                                                    # explicit option: REBEL paper Eq. 22 reward shaping
            shaping = kl_shaping(old_sum, ref_sum, torch.tensor([len(t["completion_ids"]) for t in trajs]), self.kl_coef, self.kl_units)
            base = base - shaping
        pairs = pair_indices(len(prompts), self.G, self.seed, batch_index)
        i1 = torch.tensor([a for a, _ in pairs]); i2 = torch.tensor([c for _, c in pairs])
        target = (base[i1] - base[i2]).detach()
        if self.constrained:
            dR = (R[i1] - R[i2]); dC = (C[i1] - C[i2])
            if self.kl_coef == 0 and not torch.allclose(target, dR - float(lam_t) * dC, atol=1e-5):
                raise HardFailure("target is not dR - lambda dC")
        agg = self.primal_steps(trajs, old_sum, pairs, target)
        t_train = time.time() - t0 - t_gen - t_lp
        # post-update diagnostics (no grad): KL(pi_t || pi_{t+1}) and regression fit after the step
        post: Dict[str, Any] = {}
        if self.cfg.get("post_update_pass", True):
            new_tok = batched_token_logprobs(self.policy, trajs, self.pad, self.micro, self.device, self.precision)
            new_sum = torch.stack([x.sum() for x in new_tok])
            fit = rebel_loss(new_sum[i1], new_sum[i2], old_sum[i1], old_sum[i2], target, self.eta)
            post = {"kl_prev": kl_stats(old_tok, new_tok), "post_loss": fit["loss_value"], "post_pred_target_corr": fit["pred_target_corr"],
                    "post_pred_std": fit["pred_std"], "post_sign_agreement": fit["sign_agreement"],
                    "seq_logprob_change_mean": float((new_sum - old_sum).mean()), "seq_logprob_change_abs_max": float((new_sum - old_sum).abs().max())}
        kl0 = kl_stats(old_tok, ref_tok)                                        # KL(pi_t || pi_0) on this batch's rollouts
        # dual update AFTER the primal steps, from the pre-update rollouts' mean standardized cost (all B*G responses)
        dual_rec = None
        if self.constrained:
            dual_rec = self.dual.update(float(std["cost"].mean()))
            self.state["n_dual_updates"] += 1
            self.state["n_batches_violating"] += int(dual_rec["violation"] > 0)
            if not (np.isfinite(self.dual.lam) and self.dual.lam >= 0):
                raise HardFailure(f"broken dual state lambda={self.dual.lam}")
        self._last_trajs = trajs
        lens = np.array([t["n_completion_tokens"] for t in trajs]); trunc = np.array([t["truncated"] for t in trajs])
        self.state["n_prompts"] += len(prompts); self.state["n_trajectories"] += len(trajs); self.state["n_pairs"] += len(pairs)
        self.state["n_generated_tokens"] += int(lens.sum()); self.state["n_truncated"] += int(trunc.sum()); self.state["n_eos"] += int((~trunc).sum())
        pre_fit = rebel_loss(old_sum[i1], old_sum[i2], old_sum[i1], old_sum[i2], target, self.eta)   # at theta = theta_t: loss == mean z^2
        rec: Dict[str, Any] = {
            "batch_index": batch_index, "n_prompts": len(prompts), "n_trajectories": len(trajs), "n_pairs": len(pairs), "n_generated_tokens": int(lens.sum()),
            "completion_len_mean": float(lens.mean()), "completion_len_p50": float(np.median(lens)), "completion_len_p95": float(np.percentile(lens, 95)),
            "truncation_rate": float(trunc.mean()), "eos_rate": float(1.0 - trunc.mean()), "optimizer_steps": agg["steps"], "opt_step_total": self.state["opt_step"],
            "lr": self.scheduler.get_last_lr()[0], "loss": agg["loss"] / max(1, agg["n_pairs"]), "loss_at_theta_t": pre_fit["loss_value"],
            "target_mean": pre_fit["target_mean"], "target_std": pre_fit["target_std"], "target_sq_mean": pre_fit["target_sq_mean"], "eta": self.eta,
            "grad_norm": agg["grad_norm"], "first_pair_pred_target": [agg["first_pred"], agg["first_target"]],
            "kl_initial": kl0, **post, "old_logprob_sum_mean": float(old_sum.mean()), "ref_logprob_sum_mean": float(ref_sum.mean()),
            "kl_shaping_coef": self.kl_coef, "kl_shaping_units": self.kl_units,
            "kl_shaping_pair_abs_mean": float((shaping[i1] - shaping[i2]).abs().mean()), "pair_signal_abs_mean": float(target.abs().mean()),
            "reward_len_spearman": spearman(lens.astype(float), (std["reward"] if self.constrained else std["scalar"]).numpy()),
            **{f"style_{k}": v for k, v in style_features(texts).items()},
            "t_generate_s": t_gen, "t_logprobs_s": t_lp, "t_train_s": t_train, "t_batch_s": time.time() - t0, "memory": device_memory(self.device),
            "finite": bool(np.isfinite(agg["loss"]) and kl0["finite"] and (post.get("kl_prev", {}).get("finite", True))),
        }
        for h in self.stats:
            rec[f"{h}_std_mean"] = float(std[h].mean()); rec[f"{h}_std_std"] = float(std[h].std()); rec[f"{h}_raw_mean"] = float(raw[h].mean())
        if self.constrained:
            rec.update({"lambda_used": lam_t, "lambda": dual_rec["lambda"], "log_lambda": dual_rec.get("log_lambda"), "dual_rule": self.dual_rule,
                        "lambda_unprojected": dual_rec.get("lambda_unprojected"), "dual_projected": dual_rec.get("projected", False),
                        "cost_hat": dual_rec["c_batch"], "cost_ema": dual_rec.get("c_ema"), "d": dual_rec["d"], "violation": dual_rec["violation"],
                        "constraint_violated": bool(dual_rec["violation"] > 0),
                        "frac_batches_violating": self.state["n_batches_violating"] / max(1, self.state["n_dual_updates"]),
                        "pair_dR_abs_mean": float(dR.abs().mean()), "pair_dC_abs_mean": float(dC.abs().mean()), "pair_dR_dC_corr": spearman(dR.numpy(), dC.numpy()),
                        "cost_len_spearman": spearman(lens.astype(float), std["cost"].numpy())})
        if not rec["finite"]:
            raise HardFailure("non-finite values in the training record")
        self.batch_log.write(json.dumps(rec) + "\n"); self.batch_log.flush()
        free_cache(self.device)
        return rec

    # ------------------------------------------------------------------ monitoring / dumps / diagnostics
    @torch.no_grad()
    def monitor(self, tag: str) -> Dict[str, Any]:
        t0 = time.time()
        k_resp = int(self.cfg.get("monitor_responses_per_prompt", 4)); gen_batch = int(self.cfg.get("monitor_gen_batch", 32))
        prompts = self.monitor_prompts
        trajs_all, ptexts = [], []
        for k in range(k_resp):
            for start in range(0, len(prompts), gen_batch):
                chunk = prompts[start:start + gen_batch]
                recs = generate_completions(self.policy, self.tok, [p["prompt"] for p in chunk], 1, self.gen["max_prompt_length"], self.gen["max_completion_length"],
                                            self.device, self.gen_precision, self.gen["temperature"], self.gen["top_p"], self.gen["repetition_penalty"],
                                            seed=MONITOR_GEN_SEEDS[k % len(MONITOR_GEN_SEEDS)] * 100003 + start)
                trajs_all.extend(recs); ptexts.extend(p["prompt"] for p in chunk)
        texts = [t["text"] for t in trajs_all]
        raw, std = self.score(ptexts, texts)
        pol = batched_token_logprobs(self.policy, trajs_all, self.pad, self.micro, self.device, self.precision)
        ref = batched_token_logprobs(self.ref, trajs_all, self.pad, self.micro, self.ref_device, self.precision)
        lens = np.array([t["n_completion_tokens"] for t in trajs_all])
        m: Dict[str, Any] = {"tag": tag, "batch_index": self.state["next_batch_index"], "opt_step": self.state["opt_step"], "n_prompts": len(prompts),
                             "n_trajectories": len(trajs_all), "kl_initial": kl_stats(pol, ref), "completion_len_mean": float(lens.mean()),
                             "completion_len_p95": float(np.percentile(lens, 95)), "truncation_rate": float(np.mean([t["truncated"] for t in trajs_all])),
                             **{f"style_{k}": v for k, v in style_features(texts).items()}, "seconds": time.time() - t0}
        for h in self.stats:
            m[f"{h}_std_mean"] = float(std[h].mean()); m[f"{h}_std_std"] = float(std[h].std()); m[f"{h}_raw_mean"] = float(raw[h].mean())
        if self.constrained:
            m.update({"d": self.dual.d, "cost_std_minus_d": float(std["cost"].mean()) - self.dual.d, "lambda": self.dual.lam})
        self.monitor_history.append(m)
        (self.run_dir / "monitor").mkdir(exist_ok=True)
        write_json(self.run_dir / "monitor" / f"{tag}.json", m)
        write_jsonl(self.run_dir / "monitor" / f"{tag}_samples.jsonl",
                    ({"prompt": p, "response": t["text"], "n_completion_tokens": t["n_completion_tokens"], "truncated": t["truncated"],
                      **{f"{h}_std": float(std[h][i]) for h in self.stats}} for i, (p, t) in enumerate(zip(ptexts, trajs_all))))
        write_json(self.run_dir / "monitor_history.json", self.monitor_history)
        log.info("MONITOR %s: %s", tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k not in ("tag", "kl_initial")})
        return m

    @torch.no_grad()
    def dump_generations(self, batch_index: int) -> None:
        """Light divergence trace: n_dump monitoring prompts x 1 full-distribution sample, fixed seed, with scores and style features."""
        n = int(self.cfg.get("dump_n_prompts", 32))
        prompts = self.monitor_prompts[:n]
        recs = []
        for start in range(0, len(prompts), 16):
            chunk = prompts[start:start + 16]
            recs.extend(generate_completions(self.policy, self.tok, [p["prompt"] for p in chunk], 1, self.gen["max_prompt_length"], self.gen["max_completion_length"],
                                             self.device, self.gen_precision, self.gen["temperature"], self.gen["top_p"], self.gen["repetition_penalty"],
                                             seed=DUMP_GEN_SEED * 1000 + start))
        raw, std = self.score([p["prompt"] for p in prompts], [r["text"] for r in recs])
        (self.run_dir / "generations").mkdir(exist_ok=True)
        write_jsonl(self.run_dir / "generations" / f"batch_{batch_index:05d}.jsonl",
                    ({"batch_index": batch_index, "prompt_hash": p["prompt_hash"], "prompt": p["prompt"], "response": r["text"], "n_completion_tokens": r["n_completion_tokens"],
                      "truncated": r["truncated"], **{f"{h}_std": float(std[h][i]) for h in self.stats}, **{f"{h}_raw": float(raw[h][i]) for h in self.stats},
                      **style_features([r["text"]])} for i, (p, r) in enumerate(zip(prompts, recs))))

    @torch.no_grad()
    def diagnostic(self, tag: str, gates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        dcfg = self.cfg.get("diagnostic", {}) or {}
        n = int(dcfg.get("n_samples", 32))
        prompts = self.monitor_prompts[:n]
        recs = []
        for start in range(0, len(prompts), 16):
            chunk = prompts[start:start + 16]
            recs.extend(generate_completions(self.policy, self.tok, [p["prompt"] for p in chunk], 1, self.gen["max_prompt_length"], self.gen["max_completion_length"],
                                             self.device, self.gen_precision, self.gen["temperature"], self.gen["top_p"], self.gen["repetition_penalty"],
                                             seed=int(dcfg.get("seed", 4242)) * 1000 + start))
        tm = tail_mass_and_entropy(self.policy, recs, self.pad, self.device, self.precision, k=20)
        pol = batched_token_logprobs(self.policy, recs, self.pad, self.micro, self.device, self.precision)
        ref = batched_token_logprobs(self.ref, recs, self.pad, self.micro, self.ref_device, self.precision)
        kl = kl_stats(pol, ref)
        texts = [r["text"] for r in recs]
        try:
            with open(self.run_dir / "rollout_batches.jsonl", "r", encoding="utf-8") as f:
                last_records = [json.loads(l) for l in f.readlines()[-50:]]
        except FileNotFoundError:
            last_records = []
        lens = np.array([r["n_completion_tokens"] for r in recs]) if recs else np.zeros(1)
        m: Dict[str, Any] = {"tag": tag, "batch_index": self.state["next_batch_index"], "opt_step": self.state["opt_step"], "n_samples": len(recs),
                             "tail_mass": tm["tail_mass"], "entropy": tm["entropy"], "kl_mean": kl["kl_k3_token_mean"], "kl_initial": kl,
                             "truncation_rate": float(np.mean([r["truncated"] for r in recs])) if recs else 0.0, "completion_len_mean": float(lens.mean()),
                             **ascii_stats(texts), **{f"style_{k}": v for k, v in style_features(texts).items()},
                             "finite": bool(tm["finite"] and kl["finite"] and records_finite(last_records, keys=("loss", "kl_mean"))),
                             "lambda": self.dual.lam if self.dual else None, "d": self.dual.d if self.dual else None}
        m["gates"] = evaluate_gates(m, gates)
        (self.run_dir / "diagnostic").mkdir(exist_ok=True)
        write_json(self.run_dir / "diagnostic" / f"{tag}.json", m)
        write_jsonl(self.run_dir / "diagnostic" / f"{tag}_samples.jsonl",
                    ({"prompt": p["prompt"], "prompt_hash": p["prompt_hash"], "response": r["text"], "n_completion_tokens": r["n_completion_tokens"], "truncated": r["truncated"]}
                     for p, r in zip(prompts, recs)))
        log.info("DIAGNOSTIC %s: passed=%s tail=%.4f kl=%.4f trunc=%.3f len=%.0f ascii=%.3f therapy=%.2f", tag, m["gates"]["passed"], m["tail_mass"], m["kl_mean"],
                 m["truncation_rate"], m["completion_len_mean"], m["ascii_mean"], m["style_therapy_terms_per_100w"])
        self.policy.eval()
        return m

    @torch.no_grad()
    def collapse_probe(self, trajs: List[Dict[str, Any]], rec: Dict[str, Any]) -> Dict[str, Any]:
        sub = trajs[: int(self.cfg.get("collapse_probe_trajectories", 32))]
        tm = tail_mass_and_entropy(self.policy, sub, self.pad, self.device, self.precision, k=20)
        m = {"batch_index": rec["batch_index"], "tail_mass": tm["tail_mass"], "entropy": tm["entropy"], "kl_mean": rec["kl_initial"]["kl_k3_token_mean"],
             "truncation_rate": rec["truncation_rate"], **ascii_stats([t["text"] for t in trajs]), "finite": bool(tm["finite"] and rec["finite"])}
        m["collapse"] = collapse_check(m, self.cfg.get("collapse"))
        (self.run_dir / "diagnostic").mkdir(exist_ok=True)
        with open(self.run_dir / "diagnostic" / "collapse_probes.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(m) + "\n")
        return m

    def save_intermediate(self, batch_index: int) -> None:
        d = self.run_dir / "checkpoints" / f"batch_{batch_index:05d}"
        save_checkpoint(self.policy, self.tok, d, self.cfg.get("checkpoint_dtype", "bfloat16"),
                        {"algorithm": "rebel", "arm": self.arm, "seed": self.seed, "intermediate": True, "batch_index": batch_index, **self.state,
                         "lambda": self.dual.lam if self.dual else None})
        log.info("intermediate checkpoint saved: %s", d)

    # ------------------------------------------------------------------ main loop
    def train(self) -> Dict[str, Any]:
        self.setup()
        resumed = self.load_resume() if self.cfg.get("resume", True) else False
        t_start = time.time() - self.state["elapsed_s"]
        if not resumed and self.cfg.get("monitor_at_start", True):
            self.monitor("start")
        if not resumed and int(self.cfg.get("dump_every_batches", 0) or 0):
            self.dump_generations(0)
        eval_every, resume_every = int(self.cfg["eval_every_batches"]), int(self.cfg["resume_every_batches"])
        ckpt_every, dump_every = int(self.cfg.get("checkpoint_every_batches", 0) or 0), int(self.cfg.get("dump_every_batches", 0) or 0)
        probe_every = int(self.cfg.get("collapse_check_every", 0) or 0)
        stop_after = int(self.cfg.get("stop_after_batches") or 0)
        try:
            for bi in range(self.state["next_batch_index"], len(self.batches)):
                rec = self.rollout_batch(bi, self.batches[bi])
                self.state["next_batch_index"] = bi + 1
                self.state["elapsed_s"] = time.time() - t_start
                if (bi + 1) % int(self.cfg.get("log_every_batches", 1)) == 0:
                    keys = ("batch_index", "loss", "loss_at_theta_t", "post_loss", "post_pred_target_corr", "truncation_rate", "completion_len_mean", "lambda", "cost_hat",
                            "violation", "reward_std_mean", "cost_std_mean", "scalar_std_mean", "style_therapy_terms_per_100w", "t_batch_s")
                    msg = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k in keys}
                    msg["kl0"] = round(rec["kl_initial"]["kl_k3_token_mean"], 4)
                    if "kl_prev" in rec:
                        msg["kl_prev"] = round(rec["kl_prev"]["kl_k3_token_mean"], 5)
                    log.info("batch %d/%d %s", bi + 1, len(self.batches), msg)
                last = (bi + 1) >= len(self.batches)
                if dump_every and (bi + 1) % dump_every == 0 and not last:
                    self.dump_generations(bi + 1)
                if eval_every and (bi + 1) % eval_every == 0 and not last:
                    self.monitor(f"batch_{bi + 1:05d}")
                at = {int(x) for x in (self.cfg.get("checkpoint_at_batches") or [])}
                if (ckpt_every and (bi + 1) % ckpt_every == 0 and not last) or (bi + 1) in at:
                    self.save_intermediate(bi + 1)
                if resume_every and (bi + 1) % resume_every == 0 and not last:
                    self.save_resume()
                if probe_every and (bi + 1) % probe_every == 0 and not last and not stop_after:
                    pm = self.collapse_probe(self._last_trajs, rec)
                    if pm["collapse"]["collapsed"]:
                        self.save_resume()
                        log.error("catastrophic failure at batch %d: %s -- stopping (exit 3)", bi + 1, pm["collapse"]["reasons"])
                        self.batch_log.close()
                        raise SystemExit(3)
                if stop_after and (bi + 1) >= stop_after:
                    self.state["elapsed_s"] = time.time() - t_start
                    self.save_resume()
                    m = self.diagnostic(f"batch_{bi + 1:05d}", self.cfg.get("diagnostic_gates"))
                    self.batch_log.close()
                    summary = {"run_dir": str(self.run_dir), "arm": self.arm, "seed": self.seed, "stopped_for_diagnostic": True, "stop_after_batches": stop_after,
                               **self.state, "diagnostic": m, "gates_passed": m["gates"]["passed"], "lambda": self.dual.lam if self.dual else None}
                    write_json(self.run_dir / "diagnostic_summary.json", summary)
                    log.info("stopped after %d batches for the gate diagnostic: gates_passed=%s", stop_after, m["gates"]["passed"])
                    return summary
        except HardFailure as e:
            log.error("HARD FAILURE: %s -- saving resume state and exiting 3", e)
            try:
                self.save_resume()
            finally:
                self.batch_log.close()
            raise SystemExit(3)
        self.state["elapsed_s"] = time.time() - t_start
        final = self.monitor("final")
        final_dir = self.run_dir / "checkpoints" / "final"
        save_checkpoint(self.policy, self.tok, final_dir, self.cfg.get("checkpoint_dtype", "bfloat16"),
                        {"algorithm": "rebel", "arm": self.arm, "seed": self.seed, "prompt_epochs": 1, **self.state, "final_lambda": self.dual.lam if self.dual else None})
        self.save_resume()
        summary = {"run_dir": str(self.run_dir), "algorithm": "rebel", "arm": self.arm, "seed": self.seed, "policy_model": self.cfg["policy_model"],
                   "n_train_prompts": len(self.train_prompts), "n_rollout_batches": len(self.batches), **self.state,
                   "truncation_rate": self.state["n_truncated"] / max(1, self.state["n_trajectories"]), "wallclock_s": self.state["elapsed_s"],
                   "gpu_hours": self.state["elapsed_s"] / 3600.0 * max(self.n_gpus, 1) if self.n_gpus else None, "final_checkpoint": str(final_dir),
                   "final_monitor": final, "d_std": self.dual.d if self.dual else None, "final_lambda": self.dual.lam if self.dual else None,
                   "memory": device_memory(self.device), "resumed": resumed}
        write_json(self.run_dir / "train_summary.json", summary)
        self.batch_log.close()
        log.info("done: %s", {k: v for k, v in summary.items() if k in ("n_prompts", "n_trajectories", "n_pairs", "opt_step", "wallclock_s", "truncation_rate", "final_lambda")})
        return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    summary = REBELTrainer(cfg).train()
    if summary.get("stopped_for_diagnostic") and not summary.get("gates_passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
