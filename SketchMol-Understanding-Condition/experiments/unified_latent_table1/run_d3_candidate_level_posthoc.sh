#!/usr/bin/env bash
# CPU-only post-hoc aggregation of frozen D3 raw-20 Table1 candidates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-$REPO_DIR}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
REFERENCE="${SUCC_D3_REFERENCE_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_test_rows.csv}"
EVALUATOR="$SHARED_PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py"
SUMMARIZER="$SHARED_PROJECT_DIR/experiments/unified_latent_table1/summarize_d3_table1_aggregations.py"
SUPERVISED_DIR="${SUCC_D3_SUPERVISED_DIR:-$SHARED_PROJECT_DIR/outputs/d3_event_kernel_energy_table1_n20}"
GRPO_DIR="${SUCC_D3_GRPO_DIR:-$SHARED_PROJECT_DIR/outputs/d3_event_kernel_energy_grpo_table1_n20}"
CANDIDATE_LIMIT="${SUCC_D3_CANDIDATE_LIMIT:-20}"

for path in "$REFERENCE" "$EVALUATOR" "$SUMMARIZER"; do
  [[ -f "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 2; }
done

export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$SHARED_PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$SHARED_PROJECT_DIR/inputs/tdc_oracles/drd2_graph2graph_svc_py36.pkl}"

for path in "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing pinned oracle: $path" >&2; exit 2; }
done

evaluate_one() {
  local label="$1"
  local output_dir="$2"
  local candidates="$output_dir/d3_event_kernel_energy_table1_n20_candidates.csv"
  local candidate_metrics="$output_dir/moledit_table_metrics_candidate20"
  local any_metrics="$output_dir/moledit_table_metrics_any20"

  [[ -f "$candidates" ]] || { echo "ERROR: missing raw candidate file: $candidates" >&2; exit 2; }

  "$PYTHON_BIN" "$EVALUATOR" \
    --reference "$REFERENCE" \
    --candidates "$candidates" \
    --output-dir "$candidate_metrics" \
    --candidate-limit "$CANDIDATE_LIMIT" \
    --aggregation candidate \
    --require-exact-candidate-count \
    --model-name "$label" \
    --task-filter table1 \
    --missing-oracle-policy fail

  "$PYTHON_BIN" "$EVALUATOR" \
    --reference "$REFERENCE" \
    --candidates "$candidates" \
    --output-dir "$any_metrics" \
    --candidate-limit "$CANDIDATE_LIMIT" \
    --aggregation any \
    --require-exact-candidate-count \
    --model-name "$label" \
    --task-filter table1 \
    --missing-oracle-policy fail

  "$PYTHON_BIN" "$SUMMARIZER" \
    --candidate-json "$candidate_metrics/moledit_table_summary.json" \
    --any-json "$any_metrics/moledit_table_summary.json" \
    --output-json "$output_dir/d3_table1_aggregation_audit.json" \
    --output-markdown "$output_dir/d3_table1_aggregation_audit.md" \
    --model-name "$label" \
    --candidate-limit "$CANDIDATE_LIMIT"
}

evaluate_one "D3 supervised" "$SUPERVISED_DIR"
evaluate_one "D3 + GRPO" "$GRPO_DIR"

echo "supervised=$SUPERVISED_DIR/d3_table1_aggregation_audit.json"
echo "grpo=$GRPO_DIR/d3_table1_aggregation_audit.json"
