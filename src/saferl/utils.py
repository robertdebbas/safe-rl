"""Shared utilities: config loading, seeding, deterministic JSON I/O, hashing, device selection."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str = "saferl") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- config

def _coerce(value: str) -> Any:
    """Parse a CLI override value: yaml semantics (ints, floats, bools, null, lists)."""
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_nested(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _deep_merge(base: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a YAML config (with optional single ``base:`` inheritance, path relative to the config's
    directory or the cwd) and apply ``key.subkey=value`` overrides."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_path = cfg.pop("base", None)
    if base_path:
        cand = Path(path).parent / base_path
        base_file = cand if cand.exists() else Path(base_path)
        with open(base_file, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        base.pop("base", None)
        cfg = _deep_merge(base, cfg)
        cfg["_base_path"] = str(base_file)
    cfg["_config_path"] = str(path)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        set_nested(cfg, k.strip(), _coerce(v.strip()))
    return cfg


# --------------------------------------------------------------------------- seeding

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- json / hashing

def dumps_canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace variance, ASCII-escaped."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True, ensure_ascii=True)
        f.write("\n")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write rows as canonical JSON lines (byte-deterministic for identical inputs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(dumps_canonical(r))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- torch helpers

def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0


def percentile_stats(values: List[int]) -> Dict[str, float]:
    """p50/p90/p95/p99/max/mean/n for integer lengths (nearest-rank via numpy 'higher')."""
    if len(values) == 0:
        return {"n": 0}
    arr = np.asarray(values)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": int(np.percentile(arr, 50, method="higher")),
        "p90": int(np.percentile(arr, 90, method="higher")),
        "p95": int(np.percentile(arr, 95, method="higher")),
        "p99": int(np.percentile(arr, 99, method="higher")),
        "max": int(arr.max()),
    }
