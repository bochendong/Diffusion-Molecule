#!/usr/bin/env bash
# CPU re-score of the existing Table1 group-RL candidates as honest any@20.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SOURCE_DIR="${SUCC_TABLE1_NORANK_SOURCE_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_moledit_table1_group_rl_v1/benchmark_group_rl}"
REFERENCE_CSV="${SUCC_TABLE1_NORANK_REFERENCE_CSV:-$SOURCE_DIR/direct_smiles_table1_selected_n20.csv}"
CANDIDATE_CSV="${SUCC_TABLE1_NORANK_CANDIDATE_CSV:-$SOURCE_DIR/direct_smiles_candidate_predictions.csv}"
OUTPUT_DIR="${SUCC_TABLE1_NORANK_OUTPUT_DIR:-$SOURCE_DIR/moledit_table_metrics_any20}"
CANDIDATE_LIMIT="${SUCC_TABLE1_NORANK_CANDIDATE_LIMIT:-20}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-Unified-3MDiffusion:$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts${PYTHONPATH:+:$PYTHONPATH}"

echo "Table1 honest any@${CANDIDATE_LIMIT} re-eval"
echo "  python=$PYTHON_BIN"
echo "  reference=$REFERENCE_CSV"
echo "  candidates=$CANDIDATE_CSV"
echo "  output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$REFERENCE_CSV" \
  --candidates "$CANDIDATE_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --candidate-limit "$CANDIDATE_LIMIT" \
  --model-name "DirectSMILES-EditGroupRL-any${CANDIDATE_LIMIT}" \
  --thresholds "0.65,0.15" \
  --task-filter table1 \
  --missing-oracle-policy fail

echo "anyk_markdown=$OUTPUT_DIR/moledit_table_summary.md"
