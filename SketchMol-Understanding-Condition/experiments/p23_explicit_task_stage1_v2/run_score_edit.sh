#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported by submitter}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
TRAIN_OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_fast}"
OUT="${P23_EVAL_OUTPUT_ROOT:-$TRAIN_OUT/eval_corrected_prompts}"
REFERENCE="$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919/data/table1_expanded.reference.csv"
CANDIDATES="$OUT/generated/edit_p19.raw_k8.csv"

[[ -f "$CANDIDATES" ]] || { echo "ERROR: missing P23 edit generations" >&2; exit 2; }
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi
mkdir -p "$OUT/scored/edit"
for spec in "k1:candidate:1" "k8_candidate:candidate:8" "any8:any:8"; do
  IFS=: read -r label aggregation limit <<< "$spec"
  "$PY" "$PROJECT/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$REFERENCE" --candidates "$CANDIDATES" \
    --output-dir "$OUT/scored/edit/$label" --candidate-limit "$limit" \
    --aggregation "$aggregation" --model-name "P23-$label" \
    --task-filter table1 --missing-oracle-policy fail
done
touch "$OUT/EDIT_SCORING_COMPLETE"
echo "P23 corrected-prompt edit scoring complete: $OUT/scored/edit"
