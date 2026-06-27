#!/usr/bin/env bash
# Run group-relative RL for source-conditioned external multi-property tasks.

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
OUTPUT_DIR="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_multiproperty_group_rl_v1}"
SOURCE_FILE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SOURCE_FILE:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE:-}}"
TRAIN_SOURCE_FILE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TRAIN_SOURCE_FILE:-${SUCC_EXTERNAL_MULTIPROP_TRAIN_SOURCE_FILE:-$SOURCE_FILE}}"
EVAL_SOURCE_FILE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EVAL_SOURCE_FILE:-${SUCC_EXTERNAL_MULTIPROP_EVAL_SOURCE_FILE:-$SOURCE_FILE}}"
TRAIN_ROWS_CSV="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TRAIN_ROWS_CSV:-$OUTPUT_DIR/external_multiproperty_train_rows.csv}"
EVAL_ROWS_CSV="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EVAL_ROWS_CSV:-$OUTPUT_DIR/external_multiproperty_eval_rows.csv}"
TRAIN_SUMMARY_JSON="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TRAIN_SUMMARY_JSON:-$OUTPUT_DIR/external_multiproperty_train_rows.summary.json}"
EVAL_SUMMARY_JSON="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EVAL_SUMMARY_JSON:-$OUTPUT_DIR/external_multiproperty_eval_rows.summary.json}"
TASK_SPEC_JSON="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TASK_SPEC_JSON:-$OUTPUT_DIR/external_multiproperty_task_specs.json}"
TRAIN_FEATURES_DIR="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TRAIN_FEATURES_DIR:-$OUTPUT_DIR/train_condition_features_hf_vlm}"
EVAL_FEATURES_DIR="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EVAL_FEATURES_DIR:-$OUTPUT_DIR/eval_condition_features_hf_vlm}"
MODEL_DIR="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_group_rl}"
BENCHMARK_OUTPUT_DIR="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_external_multiproperty_group_rl}"
BENCHMARK_MODEL_DIR="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_group_rl_eval}"
PREDICTION_CSV="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_predictions.csv}"
GENERATED_PROPERTIES_CSV="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}}"
SOURCE_PROPERTIES_CSV="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}}"

SUITE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SUITE:-${SUCC_EXTERNAL_MULTIPROP_SUITE:-both}}"
TASK_SPLIT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TASK_SPLIT:-${SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT:-all}}"
TASKS="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TASKS:-${SUCC_EXTERNAL_MULTIPROP_TASKS:-}}"
if [[ -n "${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TRAIN_INPUT_SPLIT:-}" ]]; then
  TRAIN_INPUT_SPLIT="$SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TRAIN_INPUT_SPLIT"
elif [[ -n "$TRAIN_SOURCE_FILE" && -n "$EVAL_SOURCE_FILE" && "$TRAIN_SOURCE_FILE" != "$EVAL_SOURCE_FILE" ]]; then
  TRAIN_INPUT_SPLIT="all"
else
  TRAIN_INPUT_SPLIT="train"
fi
if [[ -n "${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EVAL_INPUT_SPLIT:-}" ]]; then
  EVAL_INPUT_SPLIT="$SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EVAL_INPUT_SPLIT"
elif [[ -n "$TRAIN_SOURCE_FILE" && -n "$EVAL_SOURCE_FILE" && "$TRAIN_SOURCE_FILE" != "$EVAL_SOURCE_FILE" ]]; then
  EVAL_INPUT_SPLIT="all"
else
  EVAL_INPUT_SPLIT="test,eval,valid,validation"
fi
MAX_ROWS_PER_TASK="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MAX_ROWS_PER_TASK:-${SUCC_EXTERNAL_MULTIPROP_MAX_ROWS_PER_TASK:-200}}"
SEED="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SEED:-17}"
FORCE_EXPORT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_FORCE_EXPORT:-${SUCC_EXTERNAL_MULTIPROP_FORCE_EXPORT:-0}}"
RUN_FEATURE_EXPORT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_RUN_FEATURE_EXPORT:-${SUCC_EXTERNAL_MULTIPROP_RUN_FEATURE_EXPORT:-auto}}"
RUN_TRAIN="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_RUN_TRAIN:-1}"
RUN_BENCHMARK_AFTER_TRAIN="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_RUN_BENCHMARK_AFTER_TRAIN:-1}"
RESUME_CHECKPOINT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_RESUME_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
RL_CHECKPOINT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_RL_CHECKPOINT:-$MODEL_DIR/direct_smiles_generator_rl.pt}"
CONDITION_MIXING_MODE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_CONDITION_MIXING_MODE:-append_property_program}"
DISABLE_PROPERTY_RERANK="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_DISABLE_PROPERTY_RERANK:-1}"

RL_EPOCHS="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EPOCHS:-1}"
RL_BATCH_SIZE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BATCH_SIZE:-8}"
RL_EVAL_BATCH_SIZE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_EVAL_BATCH_SIZE:-32}"
RL_LR="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_LR:-5e-7}"
RL_WEIGHT_DECAY="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_WEIGHT_DECAY:-1e-4}"
RL_GRAD_CLIP="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_GRAD_CLIP:-1.0}"
RL_ROLLOUTS_PER_PROMPT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_ROLLOUTS_PER_PROMPT:-16}"
RL_PARALLEL_SAMPLES="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_PARALLEL_SAMPLES:-4}"
RL_MAX_PARALLEL_SEQUENCES="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MAX_PARALLEL_SEQUENCES:-512}"
RL_MAX_NEW_TOKENS="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MAX_NEW_TOKENS:-100}"
RL_TEMPERATURE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TEMPERATURE:-0.85}"
RL_TOP_K="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TOP_K:-40}"
RL_TOP_P="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_TOP_P:-0.95}"
RL_REPETITION_PENALTY="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_REPETITION_PENALTY:-1.15}"
RL_NO_REPEAT_NGRAM_SIZE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_NO_REPEAT_NGRAM_SIZE:-6}"
RL_MIN_NEW_TOKENS="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MIN_NEW_TOKENS:-6}"
RL_SFT_WEIGHT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SFT_WEIGHT:-0.15}"
RL_ADVANTAGE_MODE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_ADVANTAGE_MODE:-group_zscore}"
RL_ADVANTAGE_CLIP="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_ADVANTAGE_CLIP:-3.0}"
RL_SEQUENCE_LOGPROB_REDUCTION="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_SEQUENCE_LOGPROB_REDUCTION:-mean}"
RL_REFERENCE_KL_WEIGHT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_REFERENCE_KL_WEIGHT:-0.05}"
RL_REWARD_VALID_WEIGHT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_REWARD_VALID_WEIGHT:-0.25}"
RL_REWARD_STRICT_WEIGHT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_REWARD_STRICT_WEIGHT:-2.0}"
RL_REWARD_DISTANCE_WEIGHT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_REWARD_DISTANCE_WEIGHT:-0.05}"
RL_REWARD_DISTANCE_CLIP="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_REWARD_DISTANCE_CLIP:-10.0}"
RL_REWARD_SOURCE_SIMILARITY_WEIGHT="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_REWARD_SOURCE_SIMILARITY_WEIGHT:-0.5}"
RL_REWARD_SOURCE_SIMILARITY_THRESHOLD="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_REWARD_SOURCE_SIMILARITY_THRESHOLD:-0.4}"
DEVICE="${SUCC_DEVICE:-auto}"

BENCHMARK_MAX_NEW_TOKENS="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_MAX_NEW_TOKENS:-100}"
BENCHMARK_TEMPERATURE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_TEMPERATURE:-0.85}"
BENCHMARK_TOP_K="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_TOP_K:-40}"
BENCHMARK_TOP_P="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_TOP_P:-0.95}"
BENCHMARK_NUM_SAMPLES="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_NUM_SAMPLES:-20}"
BENCHMARK_PARALLEL_SAMPLES="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_PARALLEL_SAMPLES:-4}"
BENCHMARK_MAX_PARALLEL_SEQUENCES="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_MAX_PARALLEL_SEQUENCES:-512}"
BENCHMARK_REPETITION_PENALTY="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_REPETITION_PENALTY:-1.15}"
BENCHMARK_NO_REPEAT_NGRAM_SIZE="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_NO_REPEAT_NGRAM_SIZE:-6}"
BENCHMARK_MIN_NEW_TOKENS="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_BENCHMARK_MIN_NEW_TOKENS:-6}"
MIN_SOURCE_TANIMOTO="${SUCC_EXTERNAL_MULTIPROP_GROUP_RL_MIN_SOURCE_TANIMOTO:-${SUCC_EXTERNAL_MULTIPROP_MIN_SOURCE_TANIMOTO:-0.4}}"

HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
HF_DEVICE_MAP="${SUCC_HF_DEVICE_MAP:-auto}"
HF_DTYPE="${SUCC_HF_DTYPE:-auto}"
HF_BATCH_SIZE="${SUCC_HF_BATCH_SIZE:-1}"
HF_MAX_LENGTH="${SUCC_HF_MAX_LENGTH:-2048}"
HF_RENDER_IMAGE_SIZE="${SUCC_HF_RENDER_IMAGE_SIZE:-256}"
POOLED_DIM="${SUCC_POOLED_DIM:-3584}"
NUM_QUERIES="${SUCC_NUM_QUERIES:-32}"
QUERY_DIM="${SUCC_QUERY_DIM:-256}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR" "$MODEL_DIR" "$BENCHMARK_OUTPUT_DIR" "$BENCHMARK_MODEL_DIR"

echo "External multi-property direct-SMILES group-RL benchmark"
echo "  python=$PYTHON_BIN"
echo "  train_source_file=${TRAIN_SOURCE_FILE:-none}"
echo "  eval_source_file=${EVAL_SOURCE_FILE:-none}"
echo "  output_dir=$OUTPUT_DIR"
echo "  suite=$SUITE"
echo "  task_split=$TASK_SPLIT"
echo "  tasks=${TASKS:-all}"
echo "  train_input_split=$TRAIN_INPUT_SPLIT"
echo "  eval_input_split=$EVAL_INPUT_SPLIT"
echo "  max_rows_per_task=$MAX_ROWS_PER_TASK"
echo "  resume_checkpoint=$RESUME_CHECKPOINT"
echo "  run_train=$RUN_TRAIN"
echo "  run_feature_export=$RUN_FEATURE_EXPORT"
echo "  condition_mixing_mode=$CONDITION_MIXING_MODE"
echo "  disable_property_rerank=$DISABLE_PROPERTY_RERANK"
echo "  rl_sft_weight=$RL_SFT_WEIGHT"
echo "  rl_source_similarity_weight=$RL_REWARD_SOURCE_SIMILARITY_WEIGHT"
echo "  benchmark_num_samples=$BENCHMARK_NUM_SAMPLES"

export_rows() {
  local source_file="$1"
  local output_csv="$2"
  local summary_json="$3"
  local task_spec_json="$4"
  local input_split="$5"
  if [[ "$FORCE_EXPORT" == "1" || ! -f "$output_csv" ]]; then
    if [[ -z "$source_file" || ! -f "$source_file" ]]; then
      echo "ERROR: missing external source file for $output_csv. Set train/eval source env vars." >&2
      exit 2
    fi
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_external_multiproperty_benchmark_rows.py" \
      --source-file "$source_file" \
      --output-csv "$output_csv" \
      --summary-json "$summary_json" \
      --task-spec-json "$task_spec_json" \
      --suite "$SUITE" \
      --task-split "$TASK_SPLIT" \
      --tasks "$TASKS" \
      --input-split "$input_split" \
      --max-rows-per-task "$MAX_ROWS_PER_TASK" \
      --seed "$SEED"
  fi
}

export_features() {
  local rows_csv="$1"
  local features_dir="$2"
  local should_export_features=0
  if [[ "$RUN_FEATURE_EXPORT" == "1" || "$FORCE_EXPORT" == "1" ]]; then
    should_export_features=1
  elif [[ "$RUN_FEATURE_EXPORT" == "auto" && ! -f "$features_dir/query_tokens.npy" ]]; then
    should_export_features=1
  fi
  if [[ "$should_export_features" == "1" ]]; then
    export SUCC_ENCODER=hf_vlm
    export SUCC_VARIANTS=full
    export SUCC_BASELINE_CSV="$rows_csv"
    export SUCC_OUTPUT_DIR="$features_dir"
    export SUCC_POOLED_DIM="$POOLED_DIM"
    export SUCC_NUM_QUERIES="$NUM_QUERIES"
    export SUCC_QUERY_DIM="$QUERY_DIM"
    export SUCC_HF_MODEL_NAME_OR_PATH="$HF_MODEL_NAME_OR_PATH"
    export SUCC_HF_DEVICE_MAP="$HF_DEVICE_MAP"
    export SUCC_HF_DTYPE="$HF_DTYPE"
    export SUCC_HF_BATCH_SIZE="$HF_BATCH_SIZE"
    export SUCC_HF_MAX_LENGTH="$HF_MAX_LENGTH"
    export SUCC_HF_RENDER_IMAGE_SIZE="$HF_RENDER_IMAGE_SIZE"
    bash "$PROJECT_DIR/scripts/run_condition_encoder_export.sh"
  fi
}

export_rows "$TRAIN_SOURCE_FILE" "$TRAIN_ROWS_CSV" "$TRAIN_SUMMARY_JSON" "$TASK_SPEC_JSON" "$TRAIN_INPUT_SPLIT"
export_rows "$EVAL_SOURCE_FILE" "$EVAL_ROWS_CSV" "$EVAL_SUMMARY_JSON" "$TASK_SPEC_JSON" "$EVAL_INPUT_SPLIT"
export_features "$TRAIN_ROWS_CSV" "$TRAIN_FEATURES_DIR"
export_features "$EVAL_ROWS_CSV" "$EVAL_FEATURES_DIR"

TRAIN_FEATURE_ARGS=()
if [[ -f "$TRAIN_FEATURES_DIR/query_tokens.npy" ]]; then
  TRAIN_FEATURE_ARGS+=(--condition-features-dir "$TRAIN_FEATURES_DIR")
fi
EVAL_FEATURE_ARGS=()
if [[ -f "$EVAL_FEATURES_DIR/query_tokens.npy" ]]; then
  EVAL_FEATURE_ARGS+=(--eval-condition-features-dir "$EVAL_FEATURES_DIR")
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator_rl.py" \
    --train-csv "$TRAIN_ROWS_CSV" \
    --eval-csv "$EVAL_ROWS_CSV" \
    --output-dir "$MODEL_DIR" \
    --resume-checkpoint "$RESUME_CHECKPOINT" \
    "${TRAIN_FEATURE_ARGS[@]}" \
    "${EVAL_FEATURE_ARGS[@]}" \
    --condition-mixing-mode "$CONDITION_MIXING_MODE" \
    --epochs "$RL_EPOCHS" \
    --batch-size "$RL_BATCH_SIZE" \
    --eval-batch-size "$RL_EVAL_BATCH_SIZE" \
    --lr "$RL_LR" \
    --weight-decay "$RL_WEIGHT_DECAY" \
    --grad-clip "$RL_GRAD_CLIP" \
    --rollouts-per-prompt "$RL_ROLLOUTS_PER_PROMPT" \
    --parallel-samples "$RL_PARALLEL_SAMPLES" \
    --max-parallel-sequences "$RL_MAX_PARALLEL_SEQUENCES" \
    --max-new-tokens "$RL_MAX_NEW_TOKENS" \
    --temperature "$RL_TEMPERATURE" \
    --top-k "$RL_TOP_K" \
    --top-p "$RL_TOP_P" \
    --repetition-penalty "$RL_REPETITION_PENALTY" \
    --no-repeat-ngram-size "$RL_NO_REPEAT_NGRAM_SIZE" \
    --min-new-tokens "$RL_MIN_NEW_TOKENS" \
    --sft-weight "$RL_SFT_WEIGHT" \
    --advantage-mode "$RL_ADVANTAGE_MODE" \
    --advantage-clip "$RL_ADVANTAGE_CLIP" \
    --sequence-logprob-reduction "$RL_SEQUENCE_LOGPROB_REDUCTION" \
    --reference-kl-weight "$RL_REFERENCE_KL_WEIGHT" \
    --reward-valid-weight "$RL_REWARD_VALID_WEIGHT" \
    --reward-strict-weight "$RL_REWARD_STRICT_WEIGHT" \
    --reward-distance-weight "$RL_REWARD_DISTANCE_WEIGHT" \
    --reward-distance-clip "$RL_REWARD_DISTANCE_CLIP" \
    --reward-source-similarity-weight "$RL_REWARD_SOURCE_SIMILARITY_WEIGHT" \
    --reward-source-similarity-threshold "$RL_REWARD_SOURCE_SIMILARITY_THRESHOLD" \
    --seed "$SEED" \
    --device "$DEVICE"
else
  echo "Skipping external group-RL training (SUCC_EXTERNAL_MULTIPROP_GROUP_RL_RUN_TRAIN=0)"
fi

if [[ ! -f "$RL_CHECKPOINT" ]]; then
  echo "ERROR: RL checkpoint not found: $RL_CHECKPOINT" >&2
  exit 1
fi
echo "  rl_checkpoint=$RL_CHECKPOINT"

if [[ "$RUN_BENCHMARK_AFTER_TRAIN" == "1" ]]; then
  BENCHMARK_FEATURE_ARGS=()
  if [[ -f "$EVAL_FEATURES_DIR/query_tokens.npy" ]]; then
    BENCHMARK_FEATURE_ARGS+=(--condition-features-dir "$TRAIN_FEATURES_DIR" --eval-condition-features-dir "$EVAL_FEATURES_DIR")
  fi
  TRAIN_CMD=(
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py"
    --eval-only
    --eval-csv "$EVAL_ROWS_CSV"
    --resume-checkpoint "$RL_CHECKPOINT"
    "${BENCHMARK_FEATURE_ARGS[@]}"
    --condition-mixing-mode "$CONDITION_MIXING_MODE"
    --output-dir "$BENCHMARK_MODEL_DIR"
    --prediction-csv "$PREDICTION_CSV"
    --eval-batch-size "$RL_EVAL_BATCH_SIZE"
    --max-new-tokens "$BENCHMARK_MAX_NEW_TOKENS"
    --temperature "$BENCHMARK_TEMPERATURE"
    --top-k "$BENCHMARK_TOP_K"
    --top-p "$BENCHMARK_TOP_P"
    --num-samples "$BENCHMARK_NUM_SAMPLES"
    --parallel-samples "$BENCHMARK_PARALLEL_SAMPLES"
    --max-parallel-sequences "$BENCHMARK_MAX_PARALLEL_SEQUENCES"
    --repetition-penalty "$BENCHMARK_REPETITION_PENALTY"
    --no-repeat-ngram-size "$BENCHMARK_NO_REPEAT_NGRAM_SIZE"
    --min-new-tokens "$BENCHMARK_MIN_NEW_TOKENS"
    --seed "$SEED"
    --device "$DEVICE"
  )
  if [[ "$DISABLE_PROPERTY_RERANK" == "1" ]]; then
    TRAIN_CMD+=(--disable-property-rerank)
  fi
  "${TRAIN_CMD[@]}"

  EVAL_ARGS=()
  if [[ -n "$GENERATED_PROPERTIES_CSV" ]]; then
    EVAL_ARGS+=(--generated-properties-csv "$GENERATED_PROPERTIES_CSV")
  fi
  if [[ -n "$SOURCE_PROPERTIES_CSV" ]]; then
    EVAL_ARGS+=(--source-properties-csv "$SOURCE_PROPERTIES_CSV")
  fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
    --prediction-csv "$PREDICTION_CSV" \
    --output-dir "$BENCHMARK_OUTPUT_DIR" \
    --smiles-column generated_smiles \
    --source-smiles-column source_smiles \
    --min-source-tanimoto "$MIN_SOURCE_TANIMOTO" \
    --report-title "SUCC Direct SMILES External Multi-property Group-RL Benchmark" \
    "${EVAL_ARGS[@]}"

  echo
  echo "External multi-property group-RL benchmark ready:"
  echo "  train_rows=$TRAIN_ROWS_CSV"
  echo "  eval_rows=$EVAL_ROWS_CSV"
  echo "  checkpoint=$RL_CHECKPOINT"
  echo "  predictions=$PREDICTION_CSV"
  echo "  report=$BENCHMARK_OUTPUT_DIR/external_multiproperty_report.md"
  echo "  summary=$BENCHMARK_OUTPUT_DIR/external_multiproperty_summary.csv"
fi
