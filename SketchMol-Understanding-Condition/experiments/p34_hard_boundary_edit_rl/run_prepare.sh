#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P34_SCRIPT_DIR:?P34_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P34_OUTPUT_ROOT:-$PROJECT/outputs/p34_hard_boundary_edit_rl/seed_34001}"
P324="$PROJECT/outputs/p32_4_edit_specialist_source_rl/seed_32401"
mkdir -p "$OUT/eval/step-000"
cp "$P324/eval/step-000/summary.json" "$OUT/eval/step-000/summary.json"
cp "$P324/eval/step-000/candidates.jsonl" "$OUT/eval/step-000/candidates.jsonl"
printf '%s\n' "$P324/eval/step-000" > "$OUT/eval/step-000/BASELINE_REUSED_FROM_P32_4"
