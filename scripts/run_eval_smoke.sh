#!/usr/bin/env bash
# Engineering smoke of the evaluation pipeline on this machine: tiny synthetic LlamaForScore-layout evaluators
# (random weights, real Beaver tokenizer), 0.5B base policy, 8 prompts; 1-shard and 2-shard runs must give identical results.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
test -f artifacts/smoke/length_limits.json || python3 scripts/prepare_data.py --config configs/data_smoke.yaml
python3 -c "from saferl.eval_common import build_eval_manifest; print(build_eval_manifest('artifacts/smoke'))"
python3 scripts/make_tiny_evaluators.py --out runs/eval_smoke/tiny_evaluators
rm -rf runs/eval_smoke/base
CONFIG=configs/eval_smoke.yaml bash scripts/run_eval_shard.sh base 0 1 0
CONFIG=configs/eval_smoke.yaml bash scripts/run_eval_merge.sh
cp runs/eval_smoke/summary.json runs/eval_smoke/summary_1shard.json
rm -rf runs/eval_smoke/base
CONFIG=configs/eval_smoke.yaml bash scripts/run_eval_shard.sh base 0 2 0
CONFIG=configs/eval_smoke.yaml bash scripts/run_eval_shard.sh base 0 2 1
CONFIG=configs/eval_smoke.yaml bash scripts/run_eval_merge.sh
python3 - <<'PY'
import json
a=json.load(open("runs/eval_smoke/summary_1shard.json"))["results"][0]; b=json.load(open("runs/eval_smoke/summary.json"))["results"][0]
for m in ("helpfulness","harmfulness"):
    assert a[m]["seed_means"]==b[m]["seed_means"], (m, a[m]["seed_means"], b[m]["seed_means"])
    assert (a[m]["point"],a[m]["ci_low"],a[m]["ci_high"])==(b[m]["point"],b[m]["ci_low"],b[m]["ci_high"])
print("1-shard and 2-shard results identical:", a["helpfulness"]["seed_means"], a["harmfulness"]["seed_means"])
PY
