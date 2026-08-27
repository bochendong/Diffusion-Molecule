#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P26_SCRIPT_DIR:?P26_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P26_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P26_OUTPUT_ROOT:-$PROJECT/outputs/p26_decoupled_joint_rl/seed_26001}"
SPLIT="${P26_GATE_SPLIT:?P26_GATE_SPLIT must be exported}"
case "$SPLIT" in dev|final) ;; *) echo "ERROR: unsupported P26 gate split: $SPLIT" >&2; exit 2 ;; esac
for path in "$OUT/GATE_${SPLIT}_baseline_COMPLETE" "$OUT/GATE_${SPLIT}_p26_COMPLETE"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P26 gate marker: $path" >&2; exit 2; }
done
"$PY" "$PROJECT/experiments/p25_p23_joint_group_rl/compare_gate.py" \
  --baseline "$OUT/gate/$SPLIT/baseline/summary.json" \
  --rl "$OUT/gate/$SPLIT/p26/summary.json" \
  --output "$OUT/gate/$SPLIT/comparison.json"
touch "$OUT/GATE_${SPLIT}_COMPLETE"
