#!/usr/bin/env python3
"""Why do GRPO checkpoints look fine under HF generate but degenerate under full-distribution sampling?
For each checkpoint and the first N eval prompts: generate with (a) full sampling via the evaluation sampler,
(b) HF generate exactly as training used it (top_k=20 leaked from generation_config), (c) HF generate with top_k=0,
(d) greedy. Report lengths/truncation and, from a teacher-forced pass over (b)'s samples, the policy's average
probability mass OUTSIDE its top-20 tokens ("tail mass") and mean entropy. Writes runs/grpo_tail_diagnostic.json."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import saferl  # noqa
from saferl.eval_common import load_eval_prompts, prompt_seed
from saferl.eval_generate import sample_with_prompt_seeds
from saferl.grpo_rollout import encode_prompts_left_padded, eos_ids, collate_trajectories
from saferl.modeling import autocast_ctx, load_model, load_tokenizer
from saferl.utils import write_json

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoints", nargs="+", default=["Qwen/Qwen2.5-1.5B-Instruct", "runs/grpo_scalar_qwen2.5-1.5b_seed42/checkpoints/final",
                                                    "runs/grpo_shared_constrained_qwen2.5-1.5b_seed42/checkpoints/final",
                                                    "runs/grpo_separate_constrained_qwen2.5-1.5b_seed42/checkpoints/final"])
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--out", default="runs/grpo_tail_diagnostic.json")
args = ap.parse_args()
dev = "cuda"
tok = load_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
prompts = [p["prompt"] for p in load_eval_prompts("artifacts")[: args.n]]
seeds = [prompt_seed(20261030, p["prompt_hash"]) for p in load_eval_prompts("artifacts")[: args.n]]
stop = set(eos_ids(tok))
report = {}

def hf_gen(model, **kw):
    enc = encode_prompts_left_padded(tok, prompts, 512, dev)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    with autocast_ctx(dev, "bf16"):
        out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=512, eos_token_id=list(stop),
                             pad_token_id=tok.pad_token_id, use_cache=True, **kw)
    L = enc["input_ids"].shape[1]; recs = []
    for row in out[:, L:].cpu().tolist():
        comp = []; trunc = True
        for t in row:
            comp.append(t)
            if t in stop: trunc = False; break
        recs.append({"n": len(comp), "truncated": trunc, "text": tok.decode([t for t in comp if t not in stop], skip_special_tokens=True), "ids": comp})
    return recs, enc["prompt_ids"]

def summarize(recs):
    n = np.array([r["n"] for r in recs]); txt = [r["text"] for r in recs]
    ascii_frac = float(np.mean([sum(c.isascii() for c in t) / max(1, len(t)) for t in txt]))
    return {"len_mean": float(n.mean()), "len_p50": float(np.median(n)), "truncated": float(np.mean([r["truncated"] for r in recs])),
            "ascii_fraction": ascii_frac, "samples": [t[:160] for t in txt[:2]]}

for ck in args.checkpoints:
    t0 = time.time()
    m = load_model(ck, dev, torch.bfloat16); m.eval()
    r = {}
    full = sample_with_prompt_seeds(m, tok, prompts, seeds, 512, 512, dev, "bf16", 1.0, 1.0)
    r["eval_sampler_full"] = summarize([{"n": x["n_completion_tokens"], "truncated": x["truncated"], "text": x["response"]} for x in full])
    topk, pids = hf_gen(m, do_sample=True, temperature=1.0, top_p=1.0, repetition_penalty=1.0)        # as training (top_k leaks from config)
    r["hf_generate_as_training(top_k=config)"] = summarize(topk)
    r["hf_generate_top_k_0"] = summarize(hf_gen(m, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)[0])
    r["greedy"] = summarize(hf_gen(m, do_sample=False)[0])
    # tail mass under the policy at the positions of its own top-k samples
    trajs = [{"prompt_ids": pids[i], "completion_ids": topk[i]["ids"]} for i in range(len(prompts))]
    b = collate_trajectories(trajs, tok.pad_token_id)
    with torch.no_grad(), autocast_ctx(dev, "bf16"):
        logits = m(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev)).logits.float()
    probs = torch.softmax(logits[:, :-1], -1); mask = b["loss_mask"][:, 1:].to(dev).bool()
    top20 = torch.topk(probs, 20, dim=-1).values.sum(-1)
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
    r["tail_mass_outside_top20_mean"] = float((1 - top20)[mask].mean()); r["entropy_mean_nats"] = float(ent[mask].mean())
    r["seconds"] = time.time() - t0
    report[ck] = r
    print(ck, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "samples"}) for k, v in r.items()}, indent=None)[:900], flush=True)
    del m; torch.cuda.empty_cache()
write_json(args.out, report)
