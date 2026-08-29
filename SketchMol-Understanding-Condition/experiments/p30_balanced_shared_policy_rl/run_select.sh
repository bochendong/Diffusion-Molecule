#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P30_SCRIPT_DIR:?P30_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P30_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P30_OUTPUT_ROOT:-$PROJECT/outputs/p30_balanced_shared_policy_rl/seed_30001}"
"$PY" "$SCRIPT_DIR/select_dev_checkpoint.py" \
  --comparison-dir "$OUT/gate/dev" \
  --model-dir "$OUT/model/balanced_shared_rl" \
  --output "$OUT/gate/dev/selection.json" \
  --steps 10 20 30 40 50 60
touch "$OUT/DEV_SELECTION_COMPLETE"
