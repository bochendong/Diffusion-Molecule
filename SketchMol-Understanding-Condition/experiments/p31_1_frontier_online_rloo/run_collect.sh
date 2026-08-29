#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P311_SCRIPT_DIR:?P311_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P311_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P311_OUTPUT_ROOT:-$PROJECT/outputs/p31_1_frontier_online_rloo/seed_31101}"
P303="$PROJECT/outputs/p30_3_joint_multinegative_refinement/seed_30301"
"$PY" "$SCRIPT_DIR/collect_joint_gate.py" \
  --output-root "$OUT" \
  --edit-baseline "$P303/edit_eval/gate/final/baseline/summary.json" \
  --steps 25,50,100
touch "$OUT/JOINT_GATE_COMPLETE"
