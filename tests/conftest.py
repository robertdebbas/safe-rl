import os
import random
from pathlib import Path

import pytest

import saferl  # noqa: F401  (pins transformers to the torch backend before any transformers import)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def make_pair(i, official_split="train", prompt=None, better=0, safer=0, source="Alpaca3-70B", sev=(0, 2),
              safe=(True, False), cats0=(), cats1=("Violence",), r0=None, r1=None):
    from saferl.data import row_to_pair, HARM_CATEGORIES

    prompt = f"prompt number {i}" if prompt is None else prompt
    row = {
        "prompt": prompt,
        "response_0": f"response zero for {i}" if r0 is None else r0,
        "response_1": f"response one for {i}" if r1 is None else r1,
        "prompt_source": source, "response_0_source": "Alpaca-7B", "response_1_source": "Alpaca-7B",
        "is_response_0_safe": safe[0], "is_response_1_safe": safe[1],
        "response_0_harm_category": {c: c in cats0 for c in HARM_CATEGORIES},
        "response_1_harm_category": {c: c in cats1 for c in HARM_CATEGORIES},
        "response_0_severity_level": sev[0], "response_1_severity_level": sev[1],
        "better_response_id": better, "safer_response_id": safer,
        "response_0_sha256": f"h0{i}", "response_1_sha256": f"h1{i}",
    }
    return row_to_pair(row, official_split, i)


def synthetic_pool(n, official_split, seed, conflict_frac=0.25, n_prompts=None, start=0):
    """Deterministic synthetic pool with prompt groups of varying size and mixed strata."""
    rng = random.Random(seed)
    n_prompts = n_prompts or max(1, int(n * 0.6))
    sources = ["Alpaca3-70B", "Beavertails", "WizardLM-30B-Uncensored"]
    pairs = []
    for i in range(start, start + n):
        pid = rng.randrange(n_prompts)
        conflict = rng.random() < conflict_frac
        better = rng.randrange(2)
        safer = 1 - better if conflict else better
        safe = (rng.random() < 0.5, rng.random() < 0.5)
        sev = (0 if safe[0] else rng.randrange(1, 4), 0 if safe[1] else rng.randrange(1, 4))
        pairs.append(make_pair(i, official_split, prompt=f"{official_split} prompt {pid}", better=better, safer=safer,
                               source=rng.choice(sources), sev=sev, safe=safe,
                               cats1=("Violence",) if not safe[1] else ()))
    return pairs


@pytest.fixture(scope="session")
def tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture(scope="session")
def tiny_model(tokenizer):
    """Random-init tiny Qwen2 model sharing the real tokenizer vocabulary (no download)."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


@pytest.fixture(scope="session")
def artifacts_dir():
    if not (ARTIFACTS / "length_limits.json").exists():
        pytest.skip("artifacts/ not generated (run scripts/prepare_data.py)")
    return ARTIFACTS


@pytest.fixture(scope="session")
def smoke_artifacts_dir():
    if not (ARTIFACTS / "smoke" / "length_limits.json").exists():
        pytest.skip("artifacts/smoke not generated (run scripts/prepare_data.py --config configs/data_smoke.yaml)")
    return ARTIFACTS / "smoke"
