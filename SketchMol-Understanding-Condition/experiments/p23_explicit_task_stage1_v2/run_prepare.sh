#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P23_SCRIPT_DIR:?P23_SCRIPT_DIR must be exported by submitter}"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"
PY="${P23_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
OUT="${P23_OUTPUT_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323}"
DENOVO="${P23_DENOVO_CSV:-$PROJECT/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_train_rows.csv}"
DENOVO_FALLBACK="${P23_DENOVO_FALLBACK_CSV:-}"
EDIT="${P23_EDIT_CSV:-$DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
EDIT_EVAL="${P23_EDIT_EVAL_CSV:-$DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
ROWS_PER_MODE="${P23_ROWS_PER_MODE:-12000}"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi
for path in "$DENOVO" "$EDIT" "$EDIT_EVAL"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P23 input: $path" >&2; exit 2; }
done
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/data"

heldout=(--heldout-csv "$EDIT_EVAL")
for optional in \
  "$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919/data/table1_expanded.reference.csv" \
  "$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919/data/denovo_expanded.reference.csv" \
  "$PROJECT/outputs/p20_frozen_denovo_2p4p_table/seed_2020_r2/data/reference.csv"; do
  [[ -f "$optional" ]] && heldout+=(--heldout-csv "$optional")
done

if [[ -n "$DENOVO_FALLBACK" ]]; then
  [[ -f "$DENOVO_FALLBACK" ]] || { echo "ERROR: missing P23 de-novo fallback: $DENOVO_FALLBACK" >&2; exit 2; }
  augmented="$OUT/data/denovo_augmented_pool.csv"
  "$PY" "$SCRIPT_DIR/augment_denovo_pool.py" \
    --primary "$DENOVO" --fallback "$DENOVO_FALLBACK" \
    "${heldout[@]/--heldout-csv/--heldout}" --output "$augmented"
  DENOVO="$augmented"
fi

alignment=()
if [[ "${P23_ORACLE_ALIGNED_PAPER_EDITS:-0}" == "1" ]]; then
  ORACLE_DIR="${P23_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
  export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
  export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
  for path in "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
    [[ -f "$path" ]] || { echo "ERROR: missing P23 assay oracle: $path" >&2; exit 2; }
  done
  alignment=(
    --oracle-aligned-paper-edits
    --paper-rows-per-task "${P23_PAPER_ROWS_PER_TASK:-720}"
    --paper-candidate-pool-per-task "${P23_PAPER_CANDIDATE_POOL_PER_TASK:-10000}"
  )
fi

"$PY" "$SCRIPT_DIR/build_stage1_data.py" \
  --denovo-csv "$DENOVO" --edit-csv "$EDIT" "${heldout[@]}" \
  --output-dir "$OUT/data" --rows-per-mode "$ROWS_PER_MODE" \
  --minimum-edit-task-rows "${P23_MINIMUM_EDIT_TASK_ROWS:-32}" \
  --minimum-source-similarity "${P23_MINIMUM_SOURCE_SIMILARITY:-0.65}" \
  "${alignment[@]}" --seed 2323
"$PY" -m pytest -q "$SCRIPT_DIR/test_contract.py"
touch "$OUT/PREPARED"
echo "P23 data prepared: $OUT"
