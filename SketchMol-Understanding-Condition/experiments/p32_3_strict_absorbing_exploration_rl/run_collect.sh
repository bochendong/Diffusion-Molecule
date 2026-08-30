#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P323_SCRIPT_DIR:?P323_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P323_OUTPUT_ROOT:-$PROJECT/outputs/p32_3_strict_absorbing_exploration_rl/seed_32301}"
PY="${P323_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
"$PY" "$SCRIPT_DIR/collect_gate.py" \
  --eval-root "$OUT/eval" --output-dir "$OUT/joint_gate"
