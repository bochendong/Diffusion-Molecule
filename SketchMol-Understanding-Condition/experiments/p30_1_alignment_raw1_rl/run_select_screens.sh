#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P301_SCRIPT_DIR:?P301_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P301_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P301_OUTPUT_ROOT:-$PROJECT/outputs/p30_1_alignment_raw1_rl/seed_30101}"
for tag in step010 step020 rl; do
  test -f "$OUT/small_gate/$tag/result.json"
done
"$PY" "$SCRIPT_DIR/select_small_gate_checkpoint.py" \
  --output-root "$OUT" --output "$OUT/small_gate/selection.json"
touch "$OUT/SMALL_GATE_SELECTION_COMPLETE"

