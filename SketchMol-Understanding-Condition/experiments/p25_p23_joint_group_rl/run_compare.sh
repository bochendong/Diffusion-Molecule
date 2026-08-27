#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P25_SCRIPT_DIR:?P25_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P25_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P25_OUTPUT_ROOT:-$PROJECT/outputs/p25_p23_joint_group_rl/seed_2525}"
for path in "$OUT/GATE_BASELINE_COMPLETE" "$OUT/GATE_RL_COMPLETE" \
  "$OUT/gate/baseline/summary.json" "$OUT/gate/rl/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P25 comparison input: $path" >&2; exit 2; }
done
"$PY" "$SCRIPT_DIR/compare_gate.py" \
  --baseline "$OUT/gate/baseline/summary.json" --rl "$OUT/gate/rl/summary.json" \
  --output "$OUT/gate/comparison.json"
touch "$OUT/GATE_COMPLETE"
