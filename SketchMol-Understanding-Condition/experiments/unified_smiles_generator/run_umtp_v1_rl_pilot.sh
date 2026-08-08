#!/usr/bin/env bash
# Fast paired go/no-go pilot: UMTP policy -> short source-aware GRPO -> raw/finalizer checks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SEED="${UMTP_RL_PILOT_SEED:-7}"
EVAL_SEED="${UMTP_RL_PILOT_EVAL_SEED:-919}"
EVAL_CANDIDATES="${UMTP_RL_PILOT_EVAL_CANDIDATES:-8}"
JOINT_ROOT="${UMTP_JOINT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${UMTP_SOURCE_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
POLICY_ROOT="${UMTP_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_molecular_transformation_policy_v1}"
PILOT_ROOT="${UMTP_RL_PILOT_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_molecular_transformation_policy_rl_pilot_v1/seed_${SEED}}"
BASE_CHECKPOINT="${UMTP_RL_PILOT_BASE_CHECKPOINT:-$POLICY_ROOT/seed_${SEED}/policy/unified_smiles_generator.pt}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-SketchMol-Understanding-Condition/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"
RL_DIR="$PILOT_ROOT/rl"
RL_CHECKPOINT="$RL_DIR/unified_smiles_generator_group_rl.pt"

JOINT_TRAIN="$JOINT_ROOT/dataset/unified_joint_train_rows.csv"
JOINT_VALIDATION="$JOINT_ROOT/dataset/unified_joint_validation_rows.csv"
TRAIN_FEATURES="$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm"
VALIDATION_FEATURES="$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm"

for path in "$BASE_CHECKPOINT" "$JOINT_TRAIN" "$JOINT_VALIDATION"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
done
[[ -f "$SUCC_GSK3B_ORACLE_PATH" ]] || { echo "ERROR: missing pinned GSK3B oracle: $SUCC_GSK3B_ORACLE_PATH" >&2; exit 2; }
"$PYTHON_BIN" "$SCRIPT_DIR/legacy_gsk3b_oracle.py" verify --model "$SUCC_GSK3B_ORACLE_PATH"
for path in "$TRAIN_FEATURES" "$VALIDATION_FEATURES"; do
  [[ -d "$path" ]] || { echo "ERROR: missing required feature directory: $path" >&2; exit 2; }
done
if [[ -f "$PILOT_ROOT/umtp_rl_pilot_summary.json" && "${UMTP_RL_PILOT_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed pilot already exists: $PILOT_ROOT/umtp_rl_pilot_summary.json" >&2
  echo "Set UMTP_RL_PILOT_FORCE=1 or choose a new UMTP_RL_PILOT_OUTPUT_ROOT." >&2
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
    --denovo-high-count-min "${UMTP_RL_PILOT_HIGH_COUNT_MIN:-0}" \
    --denovo-high-count-multiplier "${UMTP_RL_PILOT_HIGH_COUNT_MULTIPLIER:-1.0}" \
    --task-mode "$task_mode" \
    --seed "$pool_seed"
}

TRAIN_POOL="$PILOT_ROOT/data/rl_train_pool.csv"
VALIDATION_POOL="$PILOT_ROOT/data/rl_validation_pool.csv"
TABLE1_POOL="$PILOT_ROOT/data/edit_validation_pool.csv"
RETENTION_POOL="$PILOT_ROOT/data/denovo_validation_pool.csv"

prepare_pool "$JOINT_TRAIN" "$TRAIN_POOL" "${UMTP_RL_PILOT_TRAIN_ROWS_PER_GROUP:-24}" 911 "${UMTP_RL_PILOT_TRAIN_TASK_MODE:-all}"
prepare_pool "$JOINT_VALIDATION" "$VALIDATION_POOL" "${UMTP_RL_PILOT_VALIDATION_ROWS_PER_GROUP:-4}" 913
prepare_pool "$JOINT_VALIDATION" "$TABLE1_POOL" "${UMTP_RL_PILOT_TABLE1_ROWS_PER_GROUP:-20}" 915 edit
prepare_pool "$JOINT_VALIDATION" "$RETENTION_POOL" "${UMTP_RL_PILOT_RETENTION_ROWS_PER_GROUP:-20}" 917 de_novo

evaluate_checkpoint() {
  local variant="$1"
  local checkpoint="$2"
  local task="$3"
  local eval_csv="$4"
  local features="$5"
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
      echo "ERROR: unsupported pilot task: $task" >&2
      exit 2
      ;;
  esac
  local output_dir="$PILOT_ROOT/eval/$variant/$task"
  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1 \
  SUCC_UNIFIED_CHECKPOINT="$checkpoint" \
  SUCC_UNIFIED_EVAL_CSV="$eval_csv" \
  SUCC_UNIFIED_EVAL_FEATURES_DIR="$features" \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$output_dir" \
  SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="$output_dir/candidate_pool" \
  SUCC_UNIFIED_BENCHMARK_TASKS="$benchmark_task" \
  SUCC_UNIFIED_CONDITION_LAYOUT=transformation \
  SUCC_UNIFIED_INPUT_MODALITY=source_or_null \
  SUCC_UNIFIED_METHOD_NAME="umtp_rl_pilot_${variant}" \
  SUCC_UNIFIED_DECODING_MODE=sample \
  SUCC_UNIFIED_NUM_SAMPLES="$EVAL_CANDIDATES" \
  SUCC_UNIFIED_MAX_CANDIDATES="$EVAL_CANDIDATES" \
  SUCC_UNIFIED_TOP_K_CANDIDATES="$EVAL_CANDIDATES" \
  SUCC_UNIFIED_CANDIDATE_BUDGETS="1,$EVAL_CANDIDATES" \
  SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
  SUCC_UNIFIED_TEMPERATURE="${UMTP_RL_PILOT_TEMPERATURE:-0.85}" \
  SUCC_UNIFIED_TOP_K="${UMTP_RL_PILOT_TOP_K:-40}" \
  SUCC_UNIFIED_TOP_P="${UMTP_RL_PILOT_TOP_P:-0.95}" \
  SUCC_UNIFIED_SMILES_GRAMMAR_CONSTRAINT="${UMTP_RL_PILOT_SMILES_GRAMMAR_CONSTRAINT:-0}" \
  SUCC_UNIFIED_SEED="$EVAL_SEED" \
  SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$reference_csv" \
  SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
  bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"
}

echo "=== UMTP short RL pilot: paired baseline ==="
evaluate_checkpoint baseline "$BASE_CHECKPOINT" table1 "$TABLE1_POOL" "$VALIDATION_FEATURES"
evaluate_checkpoint baseline "$BASE_CHECKPOINT" retention "$RETENTION_POOL" "$VALIDATION_FEATURES"

echo "=== UMTP short RL pilot: source-aware GRPO ==="
SUCC_UNIFIED_RL_TRAIN_CSV="$TRAIN_POOL" \
SUCC_UNIFIED_RL_EVAL_CSV="$VALIDATION_POOL" \
SUCC_UNIFIED_RL_OUTPUT_DIR="$RL_DIR" \
SUCC_UNIFIED_RL_RESUME_CHECKPOINT="$BASE_CHECKPOINT" \
SUCC_UNIFIED_RL_TRAIN_FEATURES_DIR="$TRAIN_FEATURES" \
SUCC_UNIFIED_RL_EVAL_FEATURES_DIR="$VALIDATION_FEATURES" \
SUCC_UNIFIED_CONDITION_LAYOUT=transformation \
SUCC_UNIFIED_INPUT_MODALITY=source_or_null \
SUCC_UNIFIED_RL_OBJECTIVE=grpo \
SUCC_UNIFIED_RL_EPOCHS=1 \
SUCC_UNIFIED_RL_BATCH_SIZE="${UMTP_RL_PILOT_BATCH_SIZE:-4}" \
SUCC_UNIFIED_RL_EVAL_BATCH_SIZE="${UMTP_RL_PILOT_EVAL_BATCH_SIZE:-8}" \
SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT="${UMTP_RL_PILOT_ROLLOUTS:-8}" \
SUCC_UNIFIED_RL_GRPO_UPDATE_EPOCHS=1 \
SUCC_UNIFIED_RL_LR="${UMTP_RL_PILOT_LR:-1e-6}" \
SUCC_UNIFIED_RL_SFT_WEIGHT="${UMTP_RL_PILOT_SFT_WEIGHT:-0.5}" \
SUCC_UNIFIED_RL_REFERENCE_KL_WEIGHT="${UMTP_RL_PILOT_REFERENCE_KL_WEIGHT:-0.1}" \
SUCC_UNIFIED_RL_REWARD_MODE=auto \
SUCC_UNIFIED_RL_REWARD_STRICT_WEIGHT=2.0 \
SUCC_UNIFIED_RL_REWARD_DISTANCE_WEIGHT=0.05 \
SUCC_UNIFIED_RL_REWARD_AGGREGATION="${UMTP_RL_PILOT_REWARD_AGGREGATION:-mean}" \
SUCC_UNIFIED_RL_REWARD_JOINT_BONUS_WEIGHT="${UMTP_RL_PILOT_REWARD_JOINT_BONUS_WEIGHT:-2.0}" \
SUCC_UNIFIED_RL_REWARD_BOTTLENECK_WEIGHT="${UMTP_RL_PILOT_REWARD_BOTTLENECK_WEIGHT:-0.5}" \
SUCC_UNIFIED_RL_REWARD_SOFTMIN_WEIGHT="${UMTP_RL_PILOT_REWARD_SOFTMIN_WEIGHT:-1.0}" \
SUCC_UNIFIED_RL_REWARD_SOFTMIN_TEMPERATURE="${UMTP_RL_PILOT_REWARD_SOFTMIN_TEMPERATURE:-0.25}" \
SUCC_UNIFIED_RL_REWARD_SOURCE_SIMILARITY_WEIGHT="${UMTP_RL_PILOT_SIMILARITY_WEIGHT:-2.0}" \
SUCC_UNIFIED_RL_REWARD_SOURCE_SIMILARITY_THRESHOLD="${UMTP_RL_PILOT_SIMILARITY_THRESHOLD:-0.65}" \
SUCC_UNIFIED_RL_REWARD_SOURCE_COPY_PENALTY="${UMTP_RL_PILOT_SOURCE_COPY_PENALTY:-0.5}" \
SUCC_UNIFIED_RL_MAX_NEW_TOKENS="${UMTP_RL_PILOT_MAX_NEW_TOKENS:-128}" \
SUCC_UNIFIED_RL_TEMPERATURE="${UMTP_RL_PILOT_TEMPERATURE:-0.85}" \
SUCC_UNIFIED_RL_TOP_K="${UMTP_RL_PILOT_TOP_K:-40}" \
SUCC_UNIFIED_RL_TOP_P="${UMTP_RL_PILOT_TOP_P:-0.95}" \
SUCC_UNIFIED_RL_PARALLEL_SAMPLES="${UMTP_RL_PILOT_ROLLOUTS:-8}" \
SUCC_UNIFIED_RL_MAX_PARALLEL_SEQUENCES=256 \
SUCC_UNIFIED_SMILES_GRAMMAR_CONSTRAINT="${UMTP_RL_PILOT_SMILES_GRAMMAR_CONSTRAINT:-0}" \
SUCC_UNIFIED_NUM_SAMPLES=1 \
SUCC_UNIFIED_TOP_K_CANDIDATES=1 \
SUCC_UNIFIED_RL_SEED="$SEED" \
bash "$SCRIPT_DIR/run_unified_smiles_generator_group_rl.sh"

[[ -f "$RL_CHECKPOINT" ]] || { echo "ERROR: missing RL checkpoint: $RL_CHECKPOINT" >&2; exit 2; }

echo "=== UMTP short RL pilot: paired post-RL evaluation ==="
evaluate_checkpoint rl "$RL_CHECKPOINT" table1 "$TABLE1_POOL" "$VALIDATION_FEATURES"
evaluate_checkpoint rl "$RL_CHECKPOINT" retention "$RETENTION_POOL" "$VALIDATION_FEATURES"

"$PYTHON_BIN" "$SCRIPT_DIR/collect_umtp_v1_rl_pilot.py" \
  --pilot-root "$PILOT_ROOT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --rl-checkpoint "$RL_CHECKPOINT" \
  --budgets "1,$EVAL_CANDIDATES" \
  --output-prefix "$PILOT_ROOT/umtp_rl_pilot"

echo "UMTP short RL pilot ready: $PILOT_ROOT/umtp_rl_pilot_report.md"
