#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P303_SCRIPT_DIR:?P303_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P303_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P303_OUTPUT_ROOT:-$PROJECT/outputs/p30_3_joint_multinegative_refinement/seed_30301}"
"$PY" "$SCRIPT_DIR/collect_joint_gate.py" \
  --denovo "$OUT/gate/denovo/result.json" \
  --edit-baseline "$OUT/edit_eval/gate/final/baseline/summary.json" \
  --edit-refined "$OUT/edit_eval/gate/final/refined/summary.json" \
  --output-dir "$OUT/joint_gate"
touch "$OUT/JOINT_GATE_COMPLETE"
