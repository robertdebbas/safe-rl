#!/usr/bin/env python3
"""Build tiny random LlamaForScore-layout checkpoints (keys `model.*` + `score_head.*`, config fields as in the
Beaver evaluators) with the REAL Beaver tokenizer, for engineering smoke tests of eval_score.py only."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import saferl  # noqa: E402,F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/eval_smoke/tiny_evaluators")
    ap.add_argument("--tokenizer", default="PKU-Alignment/beaver-7b-unified-reward")
    args = ap.parse_args()
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, LlamaConfig, LlamaModel

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    for name, score_type, seed in (("reward", "reward", 1), ("cost", "cost", 2)):
        torch.manual_seed(seed)
        cfg = LlamaConfig(vocab_size=32001, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=4, max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=32000)
        m = LlamaModel(cfg)
        head = torch.nn.Linear(64, 1, bias=True)
        d = Path(args.out) / name
        d.mkdir(parents=True, exist_ok=True)
        sd = {f"model.{k}": v.contiguous() for k, v in m.state_dict().items()}
        sd["score_head.weight"] = head.weight.detach().contiguous(); sd["score_head.bias"] = head.bias.detach().contiguous()
        save_file(sd, d / "model.safetensors")
        c = json.loads(cfg.to_json_string())
        c.update({"architectures": ["LlamaForScore"], "score_dim": 1, "score_bias": True, "score_type": score_type, "do_normalize": False})
        (d / "config.json").write_text(json.dumps(c, indent=1))
        tok.save_pretrained(d)
        print("wrote", d)


if __name__ == "__main__":
    main()
