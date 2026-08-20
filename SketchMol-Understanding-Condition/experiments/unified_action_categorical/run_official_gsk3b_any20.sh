#!/usr/bin/env bash
# Re-score an existing n=20 candidate CSV on the official 40-row GSK3B pack.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-$REPO_DIR}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
CANDIDATES="${1:?usage: run_official_gsk3b_any20.sh CANDIDATE_CSV OUTPUT_DIR MODEL_NAME}"
OUTPUT_DIR="${2:?}"
MODEL_NAME="${3:?}"
OFFICIAL_GSK3B="${SUCC_OFFICIAL_GSK3B_CSV:-$SHARED_PROJECT_DIR/outputs/direct_smiles_moledit_table1_gsk3b_n20_pilot_v1/gsk3b_pack/table1_eval_gsk3b_moledit_rows.csv}"

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts:$REPO_DIR/SketchMol-Unified-3MDiffusion${PYTHONPATH:+:$PYTHONPATH}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$SHARED_PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"

"$PYTHON_BIN" "$SCRIPT_DIR/align_official_gsk3b_candidates.py" \
  --official-reference "$OFFICIAL_GSK3B" \
  --candidates "$CANDIDATES" \
  --output-csv "$OUTPUT_DIR/official_gsk3b_n20_candidates.csv"

"$PYTHON_BIN" "$SHARED_PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$OFFICIAL_GSK3B" \
  --candidates "$OUTPUT_DIR/official_gsk3b_n20_candidates.csv" \
  --output-dir "$OUTPUT_DIR/moledit_table_metrics_any20" \
  --candidate-limit 20 \
  --model-name "$MODEL_NAME" \
  --task-filter table1 \
  --missing-oracle-policy fail

echo "official_gsk3b_metrics=$OUTPUT_DIR/moledit_table_metrics_any20/moledit_table_summary.json"
