#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P25_SCRIPT_DIR:?P25_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P25_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P25_OUTPUT_ROOT:-$PROJECT/outputs/p25_p23_joint_group_rl/seed_2525}"
for path in "$OUT/GATE_CHECKPOINT_013_COMPLETE" "$OUT/GATE_CHECKPOINT_026_COMPLETE"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P25 checkpoint diagnostic marker: $path" >&2; exit 2; }
done
"$PY" "$SCRIPT_DIR/compare_checkpoint_trajectory.py" \
  --gate-dir "$OUT/gate" --output "$OUT/gate/checkpoint_trajectory.json"
touch "$OUT/CHECKPOINT_DIAGNOSTIC_COMPLETE"
