#!/usr/bin/env bash
# Full MolEdit Table1 evaluation for the protected common-decoder GraphEditDSL checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${UMTP_GRAPH_ACTION_SEED:-7}"
SHARED_REPO_DIR="${UMTP_SHARED_REPO_DIR:-$REPO_DIR}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
JOINT_ROOT="${UMTP_JOINT_ROOT:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${UMTP_SOURCE_SUITE_ROOT:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_suite_v1}"
CHECKPOINT="${UMTP_GRAPH_ACTION_CHECKPOINT:-$SHARED_PROJECT_DIR/outputs/umtp_graph_action_protected_pilot_v1/seed_${SEED}/action_policy/umtp_graph_action_policy.pt}"
EVAL_CSV="${UMTP_GRAPH_ACTION_EVAL_CSV:-$JOINT_ROOT/dataset/table1_test_rows.csv}"
FEATURES="${UMTP_GRAPH_ACTION_EVAL_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
OUTPUT_ROOT="${UMTP_GRAPH_ACTION_FULL_ROOT:-$SHARED_PROJECT_DIR/outputs/umtp_graph_action_full_eval_v1/seed_${SEED}}"
EVAL_ROOT="$OUTPUT_ROOT/eval/action/table1"
CANDIDATE_DIR="$EVAL_ROOT/candidate_pool"
CANDIDATES="$CANDIDATE_DIR/graph_action_candidates.csv"
CANDIDATE_SUMMARY="$CANDIDATE_DIR/graph_action_summary.json"
BUDGETS="${UMTP_GRAPH_ACTION_FULL_BUDGETS:-1,8,20,64,256}"
TOP_CANDIDATES="${UMTP_GRAPH_ACTION_FULL_TOP_CANDIDATES:-256}"
RUN_RANK="${UMTP_GRAPH_ACTION_RUN_RANK:-auto}"
METHOD_NAME="${UMTP_GRAPH_ACTION_METHOD_NAME:-umtp_graph_action_policy_protected}"
REPORT_TITLE="${UMTP_GRAPH_ACTION_REPORT_TITLE:-Protected GraphEditDSL Full Table1 Evaluation}"
PROTOCOL="${UMTP_GRAPH_ACTION_PROTOCOL:-protected_graph_edit_dsl_full_table1}"

[[ -f "$CHECKPOINT" ]] || { echo "ERROR: missing action checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -f "$EVAL_CSV" ]] || { echo "ERROR: missing Table1 test CSV: $EVAL_CSV" >&2; exit 2; }
[[ -d "$FEATURES" ]] || { echo "ERROR: missing condition features: $FEATURES" >&2; exit 2; }
if [[ -f "$OUTPUT_ROOT/umtp_graph_action_full_eval_summary.json" && "${UMTP_GRAPH_ACTION_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed full evaluation already exists: $OUTPUT_ROOT/umtp_graph_action_full_eval_summary.json" >&2
  exit 2
fi

mkdir -p "$CANDIDATE_DIR"

if [[ "$RUN_RANK" == "auto" ]]; then
  if [[ -s "$CANDIDATES" && -s "$CANDIDATE_SUMMARY" ]]; then
    RUN_RANK=0
  else
    RUN_RANK=1
  fi
fi

case "$RUN_RANK" in
1)
  echo "=== Rank executable GraphEditDSL candidates on full Table1 test ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/umtp_graph_action_policy.py" rank \
    --checkpoint "$CHECKPOINT" \
    --eval-csv "$EVAL_CSV" \
    --eval-features-dir "$FEATURES" \
    --candidate-output-csv "$CANDIDATES" \
    --summary-json "$CANDIDATE_SUMMARY" \
    --condition-layout transformation \
    --site-limit "${UMTP_GRAPH_ACTION_SITE_LIMIT:-32}" \
    --max-actions-per-row "${UMTP_GRAPH_ACTION_MAX_ACTIONS:-512}" \
    --top-candidates "$TOP_CANDIDATES" \
    --score-batch-size "${UMTP_GRAPH_ACTION_SCORE_BATCH_SIZE:-256}" \
    --source-similarity-threshold 0.65 \
    --compact-output \
    --device auto
  ;;
0)
  [[ -s "$CANDIDATES" ]] || { echo "ERROR: missing reusable candidate CSV: $CANDIDATES" >&2; exit 2; }
  [[ -s "$CANDIDATE_SUMMARY" ]] || { echo "ERROR: missing reusable candidate summary: $CANDIDATE_SUMMARY" >&2; exit 2; }
  echo "=== Reuse existing GraphEditDSL candidate pool ==="
  echo "  candidates=$CANDIDATES"
  echo "  candidate_summary=$CANDIDATE_SUMMARY"
  ;;
*)
  echo "ERROR: UMTP_GRAPH_ACTION_RUN_RANK must be auto, 0, or 1 (got $RUN_RANK)" >&2
  exit 2
  ;;
esac

echo "=== Official Table1 metrics at shared candidate-prefix budgets ==="
SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=0 \
SUCC_UNIFIED_BENCHMARK_CANDIDATE_CSV="$CANDIDATES" \
SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$EVAL_ROOT" \
SUCC_UNIFIED_BENCHMARK_TASKS=moledit_table1 \
SUCC_UNIFIED_METHOD_NAME="$METHOD_NAME" \
SUCC_UNIFIED_CANDIDATE_BUDGETS="$BUDGETS" \
SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$EVAL_CSV" \
SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"

"$PYTHON_BIN" "$SCRIPT_DIR/collect_umtp_graph_action_full_eval.py" \
  --eval-root "$EVAL_ROOT" \
  --candidate-csv "$CANDIDATES" \
  --candidate-summary "$CANDIDATE_SUMMARY" \
  --budgets "$BUDGETS" \
  --checkpoint "$CHECKPOINT" \
  --method-name "$METHOD_NAME" \
  --report-title "$REPORT_TITLE" \
  --protocol "$PROTOCOL" \
  --output-prefix "$OUTPUT_ROOT/umtp_graph_action_full_eval"

echo "$REPORT_TITLE ready: $OUTPUT_ROOT/umtp_graph_action_full_eval_report.md"
