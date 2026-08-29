#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P30_SCRIPT_DIR:?P30_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P30_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P30_OUTPUT_ROOT:-$PROJECT/outputs/p30_balanced_shared_policy_rl/seed_30001}"
"$PY" "$SCRIPT_DIR/collect_result.py" \
  --selection "$OUT/gate/dev/selection.json" \
  --final-comparison "$OUT/gate/final/comparison.json" \
  --output-dir "$OUT"
touch "$OUT/EXPERIMENT_COMPLETE"
