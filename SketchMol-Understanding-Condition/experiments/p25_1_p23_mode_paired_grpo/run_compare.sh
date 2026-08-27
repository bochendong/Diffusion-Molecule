#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P251_SCRIPT_DIR:?P251_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P251_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P251_OUTPUT_ROOT:-$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125}"
SPLIT="${P251_GATE_SPLIT:?P251_GATE_SPLIT must be dev or final}"
for path in "$OUT/GATE_${SPLIT}_baseline_COMPLETE" "$OUT/GATE_${SPLIT}_rl_COMPLETE"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P25.1 gate marker: $path" >&2; exit 2; }
done
"$PY" "$PROJECT/experiments/p25_p23_joint_group_rl/compare_gate.py" \
  --baseline "$OUT/gate/$SPLIT/baseline/summary.json" \
  --rl "$OUT/gate/$SPLIT/rl/summary.json" \
  --output "$OUT/gate/$SPLIT/comparison.json"
touch "$OUT/GATE_${SPLIT}_COMPLETE"
