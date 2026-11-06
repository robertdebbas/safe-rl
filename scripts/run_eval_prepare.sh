#!/usr/bin/env bash
# Build the evaluation prompt manifest (test_representative deduplicated by normalized prompt).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
ART="${1:-artifacts}"
python3 -c "from saferl.eval_common import build_eval_manifest; import json; print(json.dumps(build_eval_manifest('$ART'), indent=1))"
