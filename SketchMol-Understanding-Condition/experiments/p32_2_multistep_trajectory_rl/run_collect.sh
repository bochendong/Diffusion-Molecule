#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P322_SCRIPT_DIR:?P322_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P322_OUTPUT_ROOT:-$PROJECT/outputs/p32_2_multistep_trajectory_rl/seed_32201}"
PY="${P322_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
"$PY" "$SCRIPT_DIR/collect_multistep_gate.py" \
  --eval-root "$OUT/eval" --output-dir "$OUT/joint_gate"
