#!/usr/bin/env bash
# Build unified train/eval CSVs from existing repo exports.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
SUITE_ROOT="${SUCC_UNIFIED_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
DATA_ROOT="${SUCC_UNIFIED_DATA_ROOT:-$SUITE_ROOT/dataset}"

DENovo_2P7P_ROOT="${SUCC_UNIFIED_2P7P_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
DENovo_OOD_ROOT="${SUCC_UNIFIED_OOD_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_ood_v2_mixed_condition}"
EXTERNAL_ROWS="${SUCC_UNIFIED_EXTERNAL_ROWS_CSV:-SketchMol-Understanding-Condition/outputs/external_mumo_graph_edit_admet_prior_v1/external_multiproperty_rows.csv}"
TABLE1_FAIR_ROOT="${SUCC_UNIFIED_TABLE1_FAIR_ROOT:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v2_fix/dataset/table1_benchmark_fair}"

MOLEDIT_TRAIN_SPLIT="${SUCC_UNIFIED_MOLEDIT_TRAIN_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
MOLEDIT_EVAL_SPLIT="${SUCC_UNIFIED_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
TABLE1_TRAIN_PACK="${SUCC_UNIFIED_TABLE1_TRAIN_PACK_DIR:-$DATA_ROOT/table1_train_pack}"
TABLE1_EVAL_PACK="${SUCC_UNIFIED_TABLE1_EVAL_PACK_DIR:-$DATA_ROOT/table1_eval_pack}"
TABLE1_TRAIN_PER_TASK="${SUCC_UNIFIED_TABLE1_TRAIN_PER_TASK:-500}"
TABLE1_EVAL_PER_TASK="${SUCC_UNIFIED_TABLE1_EVAL_PER_TASK:-100}"

TRAIN_CSV="${SUCC_UNIFIED_TRAIN_CSV:-$DATA_ROOT/unified_train_rows.csv}"
EVAL_CSV="${SUCC_UNIFIED_EVAL_CSV:-$DATA_ROOT/unified_eval_rows.csv}"
MOLEDIT_REFERENCE_CSV="${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV:-$TABLE1_EVAL_PACK/table1_moledit_rows.csv}"

mkdir -p "$DATA_ROOT" "$TABLE1_TRAIN_PACK" "$TABLE1_EVAL_PACK"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing required file: $1" >&2
    exit 2
  fi
}

require_file "$DENovo_2P7P_ROOT/denovo_2p7p_train_rows.csv"
require_file "$DENovo_2P7P_ROOT/denovo_2p7p_eval_rows.csv"
require_file "$DENovo_OOD_ROOT/denovo_ood_train_rows.csv"
require_file "$DENovo_OOD_ROOT/denovo_ood_eval_rows.csv"
require_file "$EXTERNAL_ROWS"

if [[ ! -f "$TABLE1_TRAIN_PACK/table1_benchmark_condition_rows.csv" ]]; then
  echo "=== exporting Table1 train pack (per_task=$TABLE1_TRAIN_PER_TASK) ==="
  "$PYTHON_BIN" "$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts/export_moledit_table1_benchmark_pack.py" \
    --moledit-train-split "$MOLEDIT_TRAIN_SPLIT" \
    --moledit-eval-split "$MOLEDIT_EVAL_SPLIT" \
    --output-dir "$TABLE1_TRAIN_PACK" \
    --per-task "$TABLE1_TRAIN_PER_TASK" \
    --eval-first 0
fi

if [[ ! -f "$TABLE1_EVAL_PACK/table1_moledit_rows.csv" ]]; then
  if [[ -f "$TABLE1_FAIR_ROOT/table1_moledit_rows.csv" ]]; then
    echo "=== reusing existing Table1 fair eval pack ==="
    mkdir -p "$TABLE1_EVAL_PACK"
    cp "$TABLE1_FAIR_ROOT/table1_benchmark_condition_rows.csv" "$TABLE1_EVAL_PACK/table1_benchmark_condition_rows.csv"
    cp "$TABLE1_FAIR_ROOT/table1_moledit_rows.csv" "$TABLE1_EVAL_PACK/table1_moledit_rows.csv"
  else
    echo "=== exporting Table1 eval pack (per_task=$TABLE1_EVAL_PER_TASK) ==="
    "$PYTHON_BIN" "$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts/export_moledit_table1_benchmark_pack.py" \
      --moledit-train-split "$MOLEDIT_TRAIN_SPLIT" \
      --moledit-eval-split "$MOLEDIT_EVAL_SPLIT" \
      --output-dir "$TABLE1_EVAL_PACK" \
      --per-task "$TABLE1_EVAL_PER_TASK" \
      --eval-first 1
  fi
fi

require_file "$TABLE1_TRAIN_PACK/table1_benchmark_condition_rows.csv"
require_file "$MOLEDIT_REFERENCE_CSV"

echo
echo "=== merging unified train rows ==="
"$PYTHON_BIN" "$SCRIPT_DIR/merge_unified_smiles_rows.py" \
  --output-csv "$TRAIN_CSV" \
  --input-csv "$DENovo_2P7P_ROOT/denovo_2p7p_train_rows.csv" \
  --input-csv "$DENovo_OOD_ROOT/denovo_ood_train_rows.csv" \
  --input-csv "$TABLE1_TRAIN_PACK/table1_benchmark_condition_rows.csv"

echo
echo "=== merging unified eval rows ==="
"$PYTHON_BIN" "$SCRIPT_DIR/merge_unified_smiles_rows.py" \
  --output-csv "$EVAL_CSV" \
  --input-csv "$DENovo_2P7P_ROOT/denovo_2p7p_eval_rows.csv" \
  --input-csv "$DENovo_OOD_ROOT/denovo_ood_eval_rows.csv" \
  --input-csv "$EXTERNAL_ROWS" \
  --input-csv "$TABLE1_EVAL_PACK/table1_benchmark_condition_rows.csv"

echo
echo "Unified data ready:"
echo "  train_csv=$TRAIN_CSV"
echo "  eval_csv=$EVAL_CSV"
echo "  moledit_reference_csv=$MOLEDIT_REFERENCE_CSV"
