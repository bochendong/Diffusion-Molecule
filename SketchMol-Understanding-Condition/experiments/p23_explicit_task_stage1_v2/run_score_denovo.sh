#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported by submitter}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
TRAIN_OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_fast}"
OUT="${P23_EVAL_OUTPUT_ROOT:-$TRAIN_OUT/eval_corrected_prompts}"
P17="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
EVALUATOR="$PROJECT/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi
export PYTHONPATH="$P17:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/scored/denovo"

score_one() {
  local label="$1" reference="$2" candidates="$3"
  local annotated="$OUT/scored/denovo/$label.annotated.csv"
  [[ -f "$candidates" ]] || return 0
  "$PY" "$P17/annotate_denovo.py" --reference "$reference" \
    --raw-candidates "$candidates" --output "$annotated"
  "$PY" "$EVALUATOR" --eval-csv "$reference" --candidates-csv "$annotated" \
    --output-json "$OUT/scored/denovo/$label.metrics.json" \
    --output-md "$OUT/scored/denovo/$label.report.md" --budgets 1,4,8
}

score_one "p19_6p7p" \
  "$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919/data/denovo_expanded.reference.csv" \
  "$OUT/generated/denovo_6p7p_p19.raw_k8.csv"
score_one "p20_2p4p" \
  "$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020/data/denovo_2p4p.reference.csv" \
  "$OUT/generated/denovo_2p4p_p20.raw_k8.csv"
echo "P23 corrected-prompt de-novo scoring complete: $OUT/scored/denovo"
