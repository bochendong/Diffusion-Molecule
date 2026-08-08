#!/usr/bin/env bash
# Validation-only pilot for official-instruction-aligned GraphEditDSL labels.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${UMTP_GRAPH_ACTION_V2_SEED:-7}"
BUDGETS="${UMTP_GRAPH_ACTION_V2_BUDGETS:-1,8,20}"
STAGE="${UMTP_GRAPH_ACTION_V2_STAGE:-all}"
SHARED_REPO_DIR="${UMTP_SHARED_REPO_DIR:-$REPO_DIR}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
JOINT_ROOT="${UMTP_JOINT_ROOT:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${UMTP_SOURCE_SUITE_ROOT:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_suite_v1}"
POLICY_ROOT="${UMTP_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/unified_molecular_transformation_policy_v1}"
V1_ROOT="${UMTP_GRAPH_ACTION_V1_ROOT:-$SHARED_PROJECT_DIR/outputs/umtp_graph_action_protected_pilot_v1/seed_${SEED}}"
PILOT_ROOT="${UMTP_GRAPH_ACTION_V2_ROOT:-$SHARED_PROJECT_DIR/outputs/umtp_graph_action_instruction_v2/seed_${SEED}}"
GSK3B_ORACLE="${SUCC_GSK3B_ORACLE_PATH:-$SHARED_PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_GSK3B_ORACLE_PATH="$GSK3B_ORACLE"

BASE_CHECKPOINT="${UMTP_GRAPH_ACTION_V2_BASE_CHECKPOINT:-$POLICY_ROOT/seed_${SEED}/policy/unified_smiles_generator.pt}"
V1_CHECKPOINT="${UMTP_GRAPH_ACTION_V1_CHECKPOINT:-$V1_ROOT/action_policy/umtp_graph_action_policy.pt}"
TRAIN_FEATURES="$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm"
VALIDATION_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"
SOURCE_DATA_ROOT="${UMTP_GRAPH_ACTION_V2_SOURCE_DATA_ROOT:-$V1_ROOT/data}"

for path in "$BASE_CHECKPOINT" "$V1_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing checkpoint: $path" >&2; exit 2; }
done
for path in "$TRAIN_FEATURES" "$VALIDATION_FEATURES" "$SOURCE_DATA_ROOT"; do
  [[ -d "$path" ]] || { echo "ERROR: missing directory: $path" >&2; exit 2; }
done
[[ -f "$GSK3B_ORACLE" ]] || { echo "ERROR: missing pinned legacy GSK3B oracle: $GSK3B_ORACLE" >&2; exit 2; }
"$PYTHON_BIN" "$SCRIPT_DIR/legacy_gsk3b_oracle.py" verify --model "$GSK3B_ORACLE"
if [[ -f "$PILOT_ROOT/umtp_graph_action_instruction_v2_summary.json" && "${UMTP_GRAPH_ACTION_V2_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed v2 pilot exists: $PILOT_ROOT/umtp_graph_action_instruction_v2_summary.json" >&2
  exit 2
fi

mkdir -p "$PILOT_ROOT/data" "$PILOT_ROOT/eval" "$PILOT_ROOT/action_policy"
for filename in train_pool.csv validation_pool.csv table1_pool.csv retention_pool.csv; do
  [[ -f "$SOURCE_DATA_ROOT/$filename" ]] || { echo "ERROR: missing audited pool: $SOURCE_DATA_ROOT/$filename" >&2; exit 2; }
  cp "$SOURCE_DATA_ROOT/$filename" "$PILOT_ROOT/data/$filename"
done

TRAIN_POOL="$PILOT_ROOT/data/train_pool.csv"
VALIDATION_POOL="$PILOT_ROOT/data/validation_pool.csv"
TABLE1_POOL="$PILOT_ROOT/data/table1_pool.csv"
RETENTION_POOL="$PILOT_ROOT/data/retention_pool.csv"
ACTION_TRAIN="$PILOT_ROOT/data/action_train_instruction_v2.csv"
ACTION_VALIDATION="$PILOT_ROOT/data/action_validation_instruction_v2.csv"
ACTION_ORACLE="$PILOT_ROOT/data/action_train_instruction_v2.manifest.json"
ORACLE_GATE="$PILOT_ROOT/data/action_oracle_gate.json"
V2_DIR="$PILOT_ROOT/action_policy"
V2_CHECKPOINT="$V2_DIR/umtp_graph_action_policy.pt"

if [[ "$STAGE" == "oracle" || "$STAGE" == "all" ]]; then
  echo "=== Build official-instruction action labels (all RDKit/TDC/SA properties) ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/umtp_graph_action_policy.py" prepare \
    --input-csv "$TRAIN_POOL" \
    --output-csv "$ACTION_TRAIN" \
    --manifest-json "$ACTION_ORACLE" \
    --site-limit "${UMTP_GRAPH_ACTION_V2_SITE_LIMIT:-32}" \
    --max-actions-per-row "${UMTP_GRAPH_ACTION_V2_MAX_ACTIONS:-512}" \
    --source-similarity-threshold 0.65 \
    --seed "$SEED"

  "$PYTHON_BIN" "$SCRIPT_DIR/check_umtp_graph_action_oracle_gate.py" \
    --manifest "$ACTION_ORACLE" \
    --output-json "$ORACLE_GATE" \
    --required-task GSK3B:increase \
    --min-fully-evaluable-rate "${UMTP_GRAPH_ACTION_V2_MIN_GSK_EVALUABLE:-0.95}" \
    --min-strict-reachability "${UMTP_GRAPH_ACTION_V2_MIN_GSK_REACHABILITY:-0.05}"

  "$PYTHON_BIN" "$SCRIPT_DIR/umtp_graph_action_policy.py" prepare \
    --input-csv "$VALIDATION_POOL" \
    --output-csv "$ACTION_VALIDATION" \
    --manifest-json "$PILOT_ROOT/data/action_validation_instruction_v2.manifest.json" \
    --site-limit "${UMTP_GRAPH_ACTION_V2_SITE_LIMIT:-32}" \
    --max-actions-per-row "${UMTP_GRAPH_ACTION_V2_MAX_ACTIONS:-512}" \
    --source-similarity-threshold 0.65 \
    --seed "$((SEED + 1))"
fi

if [[ "$STAGE" == "oracle" ]]; then
  echo "Instruction-aligned oracle preflight passed: $ORACLE_GATE"
  exit 0
fi

for path in "$ACTION_TRAIN" "$ACTION_VALIDATION" "$ACTION_ORACLE" "$ORACLE_GATE"; do
  [[ -f "$path" ]] || { echo "ERROR: missing oracle-stage artifact: $path" >&2; exit 2; }
done
"$PYTHON_BIN" "$SCRIPT_DIR/check_umtp_graph_action_oracle_gate.py" \
  --manifest "$ACTION_ORACLE" \
  --output-json "$ORACLE_GATE" \
  --required-task GSK3B:increase \
  --min-fully-evaluable-rate "${UMTP_GRAPH_ACTION_V2_MIN_GSK_EVALUABLE:-0.95}" \
  --min-strict-reachability "${UMTP_GRAPH_ACTION_V2_MIN_GSK_REACHABILITY:-0.05}"

echo "=== Train protected common decoder on instruction-aligned actions ==="
"$PYTHON_BIN" "$SCRIPT_DIR/umtp_graph_action_policy.py" train \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --train-csv "$ACTION_TRAIN" \
  --eval-csv "$ACTION_VALIDATION" \
  --train-features-dir "$TRAIN_FEATURES" \
  --eval-features-dir "$VALIDATION_FEATURES" \
  --output-dir "$V2_DIR" \
  --condition-layout transformation \
  --epochs "${UMTP_GRAPH_ACTION_V2_EPOCHS:-3}" \
  --batch-size "${UMTP_GRAPH_ACTION_V2_BATCH_SIZE:-64}" \
  --eval-batch-size "${UMTP_GRAPH_ACTION_V2_EVAL_BATCH_SIZE:-128}" \
  --samples-per-epoch "${UMTP_GRAPH_ACTION_V2_SAMPLES_PER_EPOCH:-4096}" \
  --lr "${UMTP_GRAPH_ACTION_V2_LR:-5e-5}" \
  --distill-weight "${UMTP_GRAPH_ACTION_V2_DISTILL_WEIGHT:-0.3}" \
  --trainable-scope source_action \
  --seed "$SEED" \
  --device auto

[[ -f "$V2_CHECKPOINT" ]] || { echo "ERROR: missing v2 checkpoint: $V2_CHECKPOINT" >&2; exit 2; }

evaluate_action_checkpoint() {
  local variant="$1"
  local checkpoint="$2"
  local output_dir="$PILOT_ROOT/eval/$variant/table1"
  local candidate_dir="$output_dir/candidate_pool"
  local candidates="$candidate_dir/graph_action_candidates.csv"
  mkdir -p "$candidate_dir"
  "$PYTHON_BIN" "$SCRIPT_DIR/umtp_graph_action_policy.py" rank \
    --checkpoint "$checkpoint" \
    --eval-csv "$TABLE1_POOL" \
    --eval-features-dir "$VALIDATION_FEATURES" \
    --candidate-output-csv "$candidates" \
    --summary-json "$candidate_dir/graph_action_summary.json" \
    --condition-layout transformation \
    --site-limit "${UMTP_GRAPH_ACTION_V2_SITE_LIMIT:-32}" \
    --max-actions-per-row "${UMTP_GRAPH_ACTION_V2_MAX_ACTIONS:-512}" \
    --top-candidates 20 \
    --score-batch-size "${UMTP_GRAPH_ACTION_V2_SCORE_BATCH_SIZE:-256}" \
    --source-similarity-threshold 0.65 \
    --compact-output \
    --device auto
  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=0 \
  SUCC_UNIFIED_BENCHMARK_CANDIDATE_CSV="$candidates" \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$output_dir" \
  SUCC_UNIFIED_BENCHMARK_TASKS=moledit_table1 \
  SUCC_UNIFIED_METHOD_NAME="umtp_graph_action_instruction_${variant}" \
  SUCC_UNIFIED_CANDIDATE_BUDGETS="$BUDGETS" \
  SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
  SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$TABLE1_POOL" \
  SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
  bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"
}

evaluate_retention_checkpoint() {
  local variant="$1"
  local checkpoint="$2"
  local output_dir="$PILOT_ROOT/eval/$variant/retention"
  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1 \
  SUCC_UNIFIED_CHECKPOINT="$checkpoint" \
  SUCC_UNIFIED_EVAL_CSV="$RETENTION_POOL" \
  SUCC_UNIFIED_EVAL_FEATURES_DIR="$VALIDATION_FEATURES" \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$output_dir" \
  SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="$output_dir/candidate_pool" \
  SUCC_UNIFIED_BENCHMARK_TASKS=denovo_2p7p \
  SUCC_UNIFIED_CONDITION_LAYOUT=transformation \
  SUCC_UNIFIED_INPUT_MODALITY=source_or_null \
  SUCC_UNIFIED_METHOD_NAME="umtp_graph_action_instruction_${variant}" \
  SUCC_UNIFIED_DECODING_MODE=sample \
  SUCC_UNIFIED_NUM_SAMPLES=20 \
  SUCC_UNIFIED_MAX_CANDIDATES=20 \
  SUCC_UNIFIED_TOP_K_CANDIDATES=20 \
  SUCC_UNIFIED_CANDIDATE_BUDGETS="$BUDGETS" \
  SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
  SUCC_UNIFIED_TEMPERATURE=0.85 \
  SUCC_UNIFIED_TOP_K=40 \
  SUCC_UNIFIED_TOP_P=0.95 \
  SUCC_UNIFIED_SEED=1401 \
  bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"
}

echo "=== Paired validation: v1 vs instruction-aligned v2 ==="
evaluate_action_checkpoint v1 "$V1_CHECKPOINT"
evaluate_action_checkpoint v2 "$V2_CHECKPOINT"
evaluate_retention_checkpoint base "$BASE_CHECKPOINT"
evaluate_retention_checkpoint v2 "$V2_CHECKPOINT"

"$PYTHON_BIN" "$SCRIPT_DIR/collect_umtp_graph_action_instruction_pilot.py" \
  --pilot-root "$PILOT_ROOT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --v1-checkpoint "$V1_CHECKPOINT" \
  --v2-checkpoint "$V2_CHECKPOINT" \
  --oracle-manifest "$ACTION_ORACLE" \
  --oracle-gate "$ORACLE_GATE" \
  --budgets "$BUDGETS" \
  --output-prefix "$PILOT_ROOT/umtp_graph_action_instruction_v2"

echo "Instruction-aligned GraphEditDSL v2 ready: $PILOT_ROOT/umtp_graph_action_instruction_v2_report.md"
