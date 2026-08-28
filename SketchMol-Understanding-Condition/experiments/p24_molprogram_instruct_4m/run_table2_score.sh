#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P24_SCRIPT_DIR:?P24_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P24_OUT="${P24_OUTPUT_ROOT:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003}"
P23_OUT="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
PY="${P24_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ORACLE_DIR="${P24_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl"
export SUCC_DRD2_ORACLE_PATH="$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl"
EXPECTED_GSK3B=cd8ee8a58beb14a924de72d6366456069dec599b514db1219198c6be08f0e1f6
EXPECTED_DRD2=dbc473fca922c834dbaee6eaba832caaff26d4f891734078fb1af359a111100f
FROZEN="$P23_OUT/eval_moledit_table1_500/data"
OUT="${P24_TABLE2_OUT:-$P24_OUT/eval_table2}"
MODEL_NAME="${P24_TABLE2_MODEL_NAME:-P24-balanced-refresh-sampled-once}"
for path in "$OUT/TABLE2_GENERATION_COMPLETE" "$FROZEN/table1_500.reference.csv" \
  "$OUT/generated/table2_500.sampled_once.csv" "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
  test -f "$path"
done
test "$(sha256sum "$SUCC_GSK3B_ORACLE_PATH" | awk '{print $1}')" = "$EXPECTED_GSK3B"
test "$(sha256sum "$SUCC_DRD2_ORACLE_PATH" | awk '{print $1}')" = "$EXPECTED_DRD2"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
mkdir -p "$OUT/results"
"$PY" "$PROJECT/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$FROZEN/table1_500.reference.csv" \
  --candidates "$OUT/generated/table2_500.sampled_once.csv" \
  --output-dir "$OUT/results" --candidate-limit 1 --aggregation candidate \
  --require-exact-candidate-count --model-name "$MODEL_NAME" \
  --task-filter table1 --missing-oracle-policy fail
touch "$OUT/TABLE2_COMPLETE"
echo "P24 Table 2 scoring complete: $OUT"
