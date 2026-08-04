#!/usr/bin/env bash
# Fast one-MIG pilot for one common decoder over SMILES plus executable GraphEditDSL.

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
POLICY_ROOT="${UMTP_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/unified_molecular_transformation_policy_v1}"
PILOT_ROOT="${UMTP_GRAPH_ACTION_ROOT:-$SHARED_PROJECT_DIR/outputs/umtp_graph_action_pilot_v1/seed_${SEED}}"

JOINT_TRAIN="$JOINT_ROOT/dataset/unified_joint_train_rows.csv"
JOINT_VALIDATION="$JOINT_ROOT/dataset/unified_joint_validation_rows.csv"
TRAIN_FEATURES="$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm"
VALIDATION_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"
BASE_CHECKPOINT="${UMTP_GRAPH_ACTION_BASE_CHECKPOINT:-$POLICY_ROOT/seed_${SEED}/policy/unified_smiles_generator.pt}"
ACTION_DIR="$PILOT_ROOT/action_policy"
ACTION_CHECKPOINT="$ACTION_DIR/umtp_graph_action_policy.pt"

for path in "$JOINT_TRAIN" "$JOINT_VALIDATION" "$BASE_CHECKPOINT"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
done
for path in "$TRAIN_FEATURES" "$VALIDATION_FEATURES"; do
  [[ -d "$path" ]] || { echo "ERROR: missing feature directory: $path" >&2; exit 2; }
done
if [[ -f "$PILOT_ROOT/umtp_graph_action_pilot_summary.json" && "${UMTP_GRAPH_ACTION_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed pilot already exists: $PILOT_ROOT/umtp_graph_action_pilot_summary.json" >&2
  exit 2
fi

mkdir -p "$PILOT_ROOT/data" "$PILOT_ROOT/eval"

prepare_pool() {
  local input_csv="$1"
  local output_csv="$2"
  local rows_per_group="$3"
  local pool_seed="$4"
  local task_mode="${5:-all}"
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_transformation_search_pool.py" \
    --input-csv "$input_csv" \
    --output-csv "$output_csv" \
    --manifest-json "${output_csv%.csv}.manifest.json" \
    --rows-per-group "$rows_per_group" \
    --task-mode "$task_mode" \
    --seed "$pool_seed"
}

TRAIN_POOL="$PILOT_ROOT/data/train_pool.csv"
VALIDATION_POOL="$PILOT_ROOT/data/validation_pool.csv"
TABLE1_POOL="$PILOT_ROOT/data/table1_pool.csv"
RETENTION_POOL="$PILOT_ROOT/data/retention_pool.csv"
ACTION_TRAIN="$PILOT_ROOT/data/action_train.csv"
ACTION_VALIDATION="$PILOT_ROOT/data/action_validation.csv"
ACTION_ORACLE="$PILOT_ROOT/data/action_train.manifest.json"
REUSE_DATA_ROOT="${UMTP_GRAPH_ACTION_REUSE_DATA_ROOT:-}"

if [[ -n "$REUSE_DATA_ROOT" ]]; then
  echo "=== Reuse audited GraphEditDSL pools and oracle labels ==="
  for filename in \
    train_pool.csv validation_pool.csv table1_pool.csv retention_pool.csv \
    action_train.csv action_validation.csv action_train.manifest.json action_validation.manifest.json; do
    [[ -f "$REUSE_DATA_ROOT/$filename" ]] || { echo "ERROR: missing reusable data: $REUSE_DATA_ROOT/$filename" >&2; exit 2; }
    cp "$REUSE_DATA_ROOT/$filename" "$PILOT_ROOT/data/$filename"
  done
else
  prepare_pool "$JOINT_TRAIN" "$TRAIN_POOL" "${UMTP_GRAPH_ACTION_TRAIN_ROWS_PER_GROUP:-48}" 1201
  prepare_pool "$JOINT_VALIDATION" "$VALIDATION_POOL" "${UMTP_GRAPH_ACTION_VALIDATION_ROWS_PER_GROUP:-8}" 1203
  prepare_pool "$JOINT_VALIDATION" "$TABLE1_POOL" "${UMTP_GRAPH_ACTION_TABLE1_ROWS_PER_GROUP:-20}" 1205 edit
  prepare_pool "$JOINT_VALIDATION" "$RETENTION_POOL" "${UMTP_GRAPH_ACTION_RETENTION_ROWS_PER_GROUP:-20}" 1207 de_novo

  echo "=== Project paired edits into executable GraphEditDSL labels ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/umtp_graph_action_policy.py" prepare \
    --input-csv "$TRAIN_POOL" \
    --output-csv "$ACTION_TRAIN" \
    --manifest-json "$ACTION_ORACLE" \
    --site-limit "${UMTP_GRAPH_ACTION_SITE_LIMIT:-32}" \
    --max-actions-per-row "${UMTP_GRAPH_ACTION_MAX_ACTIONS:-512}" \
    --seed "$SEED"
  "$PYTHON_BIN" "$SCRIPT_DIR/umtp_graph_action_policy.py" prepare \
    --input-csv "$VALIDATION_POOL" \
    --output-csv "$ACTION_VALIDATION" \
    --manifest-json "$PILOT_ROOT/data/action_validation.manifest.json" \
    --site-limit "${UMTP_GRAPH_ACTION_SITE_LIMIT:-32}" \
    --max-actions-per-row "${UMTP_GRAPH_ACTION_MAX_ACTIONS:-512}" \
    --seed "$((SEED + 1))"
fi

echo "=== Train the same decoder on de novo SMILES plus edit programs ==="
"$PYTHON_BIN" "$SCRIPT_DIR/umtp_graph_action_policy.py" train \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --train-csv "$ACTION_TRAIN" \
  --eval-csv "$ACTION_VALIDATION" \
  --train-features-dir "$TRAIN_FEATURES" \
  --eval-features-dir "$VALIDATION_FEATURES" \
  --output-dir "$ACTION_DIR" \
  --condition-layout transformation \
  --epochs "${UMTP_GRAPH_ACTION_EPOCHS:-3}" \
  --batch-size "${UMTP_GRAPH_ACTION_BATCH_SIZE:-64}" \
  --eval-batch-size "${UMTP_GRAPH_ACTION_EVAL_BATCH_SIZE:-128}" \
  --samples-per-epoch "${UMTP_GRAPH_ACTION_SAMPLES_PER_EPOCH:-4096}" \
  --lr "${UMTP_GRAPH_ACTION_LR:-5e-5}" \
  --distill-weight "${UMTP_GRAPH_ACTION_DISTILL_WEIGHT:-0.3}" \
  --trainable-scope "${UMTP_GRAPH_ACTION_TRAINABLE_SCOPE:-source_action}" \
  --seed "$SEED" \
  --device auto

[[ -f "$ACTION_CHECKPOINT" ]] || { echo "ERROR: missing action checkpoint: $ACTION_CHECKPOINT" >&2; exit 2; }

evaluate_smiles_checkpoint() {
  local variant="$1"
  local checkpoint="$2"
  local task="$3"
  local eval_csv="$4"
  local benchmark_task=""
  local reference_csv=""
  case "$task" in
    table1)
      benchmark_task=moledit_table1
      reference_csv="$eval_csv"
      ;;
    retention)
      benchmark_task=denovo_2p7p
      ;;
    *)
      echo "ERROR: unsupported task: $task" >&2
      exit 2
      ;;
  esac
  local output_dir="$PILOT_ROOT/eval/$variant/$task"
  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1 \
  SUCC_UNIFIED_CHECKPOINT="$checkpoint" \
  SUCC_UNIFIED_EVAL_CSV="$eval_csv" \
  SUCC_UNIFIED_EVAL_FEATURES_DIR="$VALIDATION_FEATURES" \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$output_dir" \
  SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="$output_dir/candidate_pool" \
  SUCC_UNIFIED_BENCHMARK_TASKS="$benchmark_task" \
  SUCC_UNIFIED_CONDITION_LAYOUT=transformation \
  SUCC_UNIFIED_INPUT_MODALITY=source_or_null \
  SUCC_UNIFIED_METHOD_NAME="umtp_graph_action_${variant}" \
  SUCC_UNIFIED_DECODING_MODE=sample \
  SUCC_UNIFIED_NUM_SAMPLES=8 \
  SUCC_UNIFIED_MAX_CANDIDATES=8 \
  SUCC_UNIFIED_TOP_K_CANDIDATES=8 \
  SUCC_UNIFIED_CANDIDATE_BUDGETS="1,8" \
  SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
  SUCC_UNIFIED_TEMPERATURE=0.85 \
  SUCC_UNIFIED_TOP_K=40 \
  SUCC_UNIFIED_TOP_P=0.95 \
  SUCC_UNIFIED_SEED=1211 \
  SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$reference_csv" \
  SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
  bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"
}

echo "=== Paired baseline ==="
evaluate_smiles_checkpoint baseline "$BASE_CHECKPOINT" table1 "$TABLE1_POOL"
evaluate_smiles_checkpoint baseline "$BASE_CHECKPOINT" retention "$RETENTION_POOL"

echo "=== Common-decoder GraphEditDSL ranking ==="
ACTION_CANDIDATES="$PILOT_ROOT/eval/action/table1/candidate_pool/graph_action_candidates.csv"
"$PYTHON_BIN" "$SCRIPT_DIR/umtp_graph_action_policy.py" rank \
  --checkpoint "$ACTION_CHECKPOINT" \
  --eval-csv "$TABLE1_POOL" \
  --eval-features-dir "$VALIDATION_FEATURES" \
  --candidate-output-csv "$ACTION_CANDIDATES" \
  --summary-json "$PILOT_ROOT/eval/action/table1/candidate_pool/graph_action_summary.json" \
  --condition-layout transformation \
  --site-limit "${UMTP_GRAPH_ACTION_SITE_LIMIT:-32}" \
  --max-actions-per-row "${UMTP_GRAPH_ACTION_MAX_ACTIONS:-512}" \
  --top-candidates 8 \
  --score-batch-size "${UMTP_GRAPH_ACTION_SCORE_BATCH_SIZE:-256}" \
  --device auto

SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=0 \
SUCC_UNIFIED_BENCHMARK_CANDIDATE_CSV="$ACTION_CANDIDATES" \
SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$PILOT_ROOT/eval/action/table1" \
SUCC_UNIFIED_BENCHMARK_TASKS=moledit_table1 \
SUCC_UNIFIED_METHOD_NAME=umtp_graph_action_policy \
SUCC_UNIFIED_CANDIDATE_BUDGETS="1,8" \
SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$TABLE1_POOL" \
SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"

evaluate_smiles_checkpoint action "$ACTION_CHECKPOINT" retention "$RETENTION_POOL"

"$PYTHON_BIN" "$SCRIPT_DIR/collect_umtp_graph_action_pilot.py" \
  --pilot-root "$PILOT_ROOT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --action-checkpoint "$ACTION_CHECKPOINT" \
  --oracle-manifest "$ACTION_ORACLE" \
  --budgets 1,8 \
  --output-prefix "$PILOT_ROOT/umtp_graph_action_pilot"

echo "UMTP common-decoder GraphEditDSL pilot ready: $PILOT_ROOT/umtp_graph_action_pilot_report.md"
