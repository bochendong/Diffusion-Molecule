#!/usr/bin/env bash
# Build train-only de novo, Table1, and MuMO action supervision for the common LLM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
OUTPUT_ROOT="${SUCC_UCA_COMMON_LLM_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1}"
DATA_DIR="$OUTPUT_ROOT/data"
MUMO_SOURCE="${SUCC_UCA_MUMO_TRAIN_JSON:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/train.json}"
JOINT_TRAIN="${SUCC_UCA_JOINT_TRAIN_CSV:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/unified_joint_train_rows.csv}"
TABLE1_ACTION_TRAIN="${SUCC_UCA_TABLE1_ACTION_TRAIN_CSV:-$PROJECT_DIR/outputs/umtp_graph_action_instruction_v2/seed_7/data/action_train_instruction_v2.csv}"
MUMO_ROWS="$DATA_DIR/mumo_train_rows.csv"
MUMO_ACTIONS="$DATA_DIR/mumo_paired_graph_actions.csv"
MUMO_ACTION_MANIFEST="$DATA_DIR/mumo_paired_graph_actions.manifest.json"
SFT_DIR="$DATA_DIR/common_llm_sft"

for path in "$MUMO_SOURCE" "$JOINT_TRAIN" "$TABLE1_ACTION_TRAIN"; do
  [[ -f "$path" ]] || { echo "ERROR: missing common-LLM data input: $path" >&2; exit 2; }
done
mkdir -p "$DATA_DIR"

echo "=== Export balanced MuMO train-only source/target pairs ==="
"$PYTHON_BIN" SketchMol-Understanding-Condition/scripts/export_external_multiproperty_benchmark_rows.py \
  --source-file "$MUMO_SOURCE" \
  --output-csv "$MUMO_ROWS" \
  --summary-json "$DATA_DIR/mumo_train_rows.summary.json" \
  --task-spec-json "$DATA_DIR/mumo_train_rows.task_specs.json" \
  --suite mumo \
  --task-split all \
  --input-split train \
  --respect-input-task \
  --max-rows-per-task "${SUCC_UCA_MUMO_ROWS_PER_TASK:-50}" \
  --seed "${SUCC_UCA_SEED:-1701}"

echo "=== Project MuMO train pairs into target-aligned GraphEditDSL actions ==="
"$PYTHON_BIN" "$SCRIPT_DIR/build_paired_graph_action_teacher.py" \
  --input-csv "$MUMO_ROWS" \
  --output-csv "$MUMO_ACTIONS" \
  --manifest-json "$MUMO_ACTION_MANIFEST" \
  --site-limit "${SUCC_UCA_SITE_LIMIT:-32}" \
  --max-actions-per-row "${SUCC_UCA_MAX_ACTIONS_PER_ROW:-512}" \
  --min-target-similarity "${SUCC_UCA_MIN_TARGET_SIMILARITY:-0.20}" \
  --min-source-similarity "${SUCC_UCA_MIN_SOURCE_SIMILARITY:-0.10}"

echo "=== Build contamination-safe common-LLM chat SFT data ==="
"$PYTHON_BIN" "$SCRIPT_DIR/build_common_llm_sft_dataset.py" \
  --denovo-train-csv "$JOINT_TRAIN" \
  --table1-action-train-csv "$TABLE1_ACTION_TRAIN" \
  --mumo-action-train-csv "$MUMO_ACTIONS" \
  --output-dir "$SFT_DIR" \
  --max-denovo-rows "${SUCC_UCA_MAX_DENOVO_ROWS:-2048}" \
  --table1-repeat "${SUCC_UCA_TABLE1_REPEAT:-3}" \
  --mumo-repeat "${SUCC_UCA_MUMO_REPEAT:-3}" \
  --validation-fraction "${SUCC_UCA_VALIDATION_FRACTION:-0.05}" \
  --seed "${SUCC_UCA_SEED:-1701}"

echo "Common-LLM train-only data ready: $SFT_DIR"
