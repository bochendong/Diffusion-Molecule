#!/usr/bin/env bash
# Validation-only pilot: video-inspired source consistency for edits plus grammar-valid de novo SMILES.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIFIED_DIR="$(cd "$SCRIPT_DIR/../unified_smiles_generator" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${P1_CONSISTENCY_SEED:-7}"
EVAL_SEED="${P1_CONSISTENCY_EVAL_SEED:-20260823}"
BUDGETS="${P1_CONSISTENCY_BUDGETS:-1,8}"
MAX_BUDGET="${P1_CONSISTENCY_MAX_BUDGET:-8}"
PROJECT_DIR="$REPO_DIR/SketchMol-Understanding-Condition"
JOINT_ROOT="${UMTP_JOINT_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${UMTP_SOURCE_SUITE_ROOT:-$PROJECT_DIR/outputs/unified_smiles_generator_suite_v1}"
PROTECTED_ROOT="${P1_CONSISTENCY_PROTECTED_ROOT:-$PROJECT_DIR/outputs/umtp_graph_action_protected_pilot_v1/seed_${SEED}}"
CHECKPOINT="${P1_CONSISTENCY_CHECKPOINT:-$PROTECTED_ROOT/action_policy/umtp_graph_action_policy.pt}"
TABLE1_POOL="${P1_CONSISTENCY_TABLE1_POOL:-$PROTECTED_ROOT/data/table1_pool.csv}"
RETENTION_POOL="${P1_CONSISTENCY_RETENTION_POOL:-$PROTECTED_ROOT/data/retention_pool.csv}"
FEATURES="${P1_CONSISTENCY_FEATURES:-$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm}"
OUTPUT_ROOT="${P1_CONSISTENCY_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p1_source_consistency_validity_pilot_v1/seed_${SEED}}"

for path in "$CHECKPOINT" "$TABLE1_POOL" "$RETENTION_POOL"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
done
[[ -d "$FEATURES" ]] || { echo "ERROR: missing validation features: $FEATURES" >&2; exit 2; }
if [[ -f "$OUTPUT_ROOT/p1_source_consistency_validity_summary.json" && "${P1_CONSISTENCY_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed pilot already exists: $OUTPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/eval"

evaluate_edit_variant() {
  local variant="$1"
  local fingerprint_weight="$2"
  local scaffold_weight="$3"
  local magnitude_weight="$4"
  local eval_root="$OUTPUT_ROOT/eval/$variant/table1"
  local candidate_dir="$eval_root/candidate_pool"
  local candidates="$candidate_dir/graph_action_candidates.csv"
  mkdir -p "$candidate_dir"

  echo "=== Edit variant: $variant ==="
  "$PYTHON_BIN" "$UNIFIED_DIR/umtp_graph_action_policy.py" rank \
    --checkpoint "$CHECKPOINT" \
    --eval-csv "$TABLE1_POOL" \
    --eval-features-dir "$FEATURES" \
    --candidate-output-csv "$candidates" \
    --summary-json "$candidate_dir/graph_action_summary.json" \
    --condition-layout transformation \
    --site-limit "${P1_CONSISTENCY_SITE_LIMIT:-32}" \
    --max-actions-per-row "${P1_CONSISTENCY_MAX_ACTIONS:-512}" \
    --top-candidates "$MAX_BUDGET" \
    --score-batch-size "${P1_CONSISTENCY_SCORE_BATCH_SIZE:-256}" \
    --source-similarity-threshold 0.65 \
    --consistency-fingerprint-weight "$fingerprint_weight" \
    --consistency-scaffold-weight "$scaffold_weight" \
    --consistency-edit-magnitude-weight "$magnitude_weight" \
    --device auto

  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=0 \
  SUCC_UNIFIED_BENCHMARK_CANDIDATE_CSV="$candidates" \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$eval_root" \
  SUCC_UNIFIED_BENCHMARK_TASKS=moledit_table1 \
  SUCC_UNIFIED_METHOD_NAME="p1_${variant}" \
  SUCC_UNIFIED_CANDIDATE_BUDGETS="$BUDGETS" \
  SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
  SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$TABLE1_POOL" \
  SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
  bash "$UNIFIED_DIR/run_unified_smiles_generator_benchmark_suite.sh"
}

evaluate_denovo_variant() {
  local variant="$1"
  local grammar="$2"
  local repetition_penalty="$3"
  local no_repeat_ngram="$4"
  local eval_root="$OUTPUT_ROOT/eval/$variant/retention"

  echo "=== De novo validity variant: $variant ==="
  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1 \
  SUCC_UNIFIED_CHECKPOINT="$CHECKPOINT" \
  SUCC_UNIFIED_EVAL_CSV="$RETENTION_POOL" \
  SUCC_UNIFIED_EVAL_FEATURES_DIR="$FEATURES" \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$eval_root" \
  SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="$eval_root/candidate_pool" \
  SUCC_UNIFIED_BENCHMARK_TASKS=denovo_2p7p \
  SUCC_UNIFIED_CONDITION_LAYOUT=transformation \
  SUCC_UNIFIED_INPUT_MODALITY=source_or_null \
  SUCC_UNIFIED_METHOD_NAME="p1_${variant}" \
  SUCC_UNIFIED_DECODING_MODE=sample \
  SUCC_UNIFIED_NUM_SAMPLES="$MAX_BUDGET" \
  SUCC_UNIFIED_MAX_CANDIDATES="$MAX_BUDGET" \
  SUCC_UNIFIED_TOP_K_CANDIDATES="$MAX_BUDGET" \
  SUCC_UNIFIED_CANDIDATE_BUDGETS="$BUDGETS" \
  SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
  SUCC_UNIFIED_TEMPERATURE="${P1_CONSISTENCY_TEMPERATURE:-0.85}" \
  SUCC_UNIFIED_TOP_K="${P1_CONSISTENCY_TOP_K:-40}" \
  SUCC_UNIFIED_TOP_P="${P1_CONSISTENCY_TOP_P:-0.95}" \
  SUCC_UNIFIED_REPETITION_PENALTY="$repetition_penalty" \
  SUCC_UNIFIED_NO_REPEAT_NGRAM_SIZE="$no_repeat_ngram" \
  SUCC_UNIFIED_MIN_NEW_TOKENS=6 \
  SUCC_UNIFIED_SMILES_GRAMMAR_CONSTRAINT="$grammar" \
  SUCC_UNIFIED_SEED="$EVAL_SEED" \
  bash "$UNIFIED_DIR/run_unified_smiles_generator_benchmark_suite.sh"
}

# Source-only weights: no target molecule and no output property oracle enters ranking.
evaluate_edit_variant policy 0.0 0.0 0.0
evaluate_edit_variant consistent \
  "${P1_CONSISTENCY_FP_WEIGHT:-1.5}" \
  "${P1_CONSISTENCY_SCAFFOLD_WEIGHT:-0.75}" \
  "${P1_CONSISTENCY_MAGNITUDE_WEIGHT:-0.05}"
evaluate_edit_variant strong_consistent \
  "${P1_CONSISTENCY_STRONG_FP_WEIGHT:-3.0}" \
  "${P1_CONSISTENCY_STRONG_SCAFFOLD_WEIGHT:-1.5}" \
  "${P1_CONSISTENCY_STRONG_MAGNITUDE_WEIGHT:-0.10}"

evaluate_denovo_variant baseline 0 1.15 6
evaluate_denovo_variant grammar_valid 1 1.05 0

"$PYTHON_BIN" "$SCRIPT_DIR/collect_p1_source_consistency_validity_pilot.py" \
  --pilot-root "$OUTPUT_ROOT" \
  --budgets "$BUDGETS" \
  --output-prefix "$OUTPUT_ROOT/p1_source_consistency_validity"

echo "P1 source-consistency + validity pilot ready: $OUTPUT_ROOT/p1_source_consistency_validity_report.md"
