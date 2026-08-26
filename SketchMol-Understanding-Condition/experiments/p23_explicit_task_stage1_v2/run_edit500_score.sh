#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
TRAIN_OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned}"
OUT="${P23_EDIT500_OUTPUT_ROOT:-$TRAIN_OUT/eval_moledit_table1_500}"
ORACLE_DIR="${P23_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
for path in "$OUT/EDIT500_GENERATED" "$OUT/data/table1_500.reference.csv" \
  "$OUT/generated/table1_500.sampled_once.csv" \
  "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P23 edit500 scoring input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
mkdir -p "$OUT/results"
"$PY" "$PROJECT/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$OUT/data/table1_500.reference.csv" \
  --candidates "$OUT/generated/table1_500.sampled_once.csv" \
  --output-dir "$OUT/results" --candidate-limit 1 --aggregation candidate \
  --require-exact-candidate-count --model-name P23-aligned24k-sampled-once \
  --task-filter table1 --missing-oracle-policy fail
touch "$OUT/EDIT500_COMPLETE"
