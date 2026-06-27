#!/usr/bin/env bash
# Run source-aware SFT warm-start for external source-conditioned multi-property tasks.

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
OUTPUT_DIR="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_mumo_source_edit_sft_v1}"
SOURCE_FILE="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_SOURCE_FILE:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE:-}}"
TRAIN_SOURCE_FILE="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TRAIN_SOURCE_FILE:-${SUCC_EXTERNAL_MULTIPROP_TRAIN_SOURCE_FILE:-$SOURCE_FILE}}"
EVAL_SOURCE_FILE="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_EVAL_SOURCE_FILE:-${SUCC_EXTERNAL_MULTIPROP_EVAL_SOURCE_FILE:-$SOURCE_FILE}}"
TRAIN_ROWS_CSV="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TRAIN_ROWS_CSV:-$OUTPUT_DIR/external_multiproperty_train_rows.csv}"
EVAL_ROWS_CSV="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_EVAL_ROWS_CSV:-$OUTPUT_DIR/external_multiproperty_eval_rows.csv}"
TRAIN_SUMMARY_JSON="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TRAIN_SUMMARY_JSON:-$OUTPUT_DIR/external_multiproperty_train_rows.summary.json}"
EVAL_SUMMARY_JSON="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_EVAL_SUMMARY_JSON:-$OUTPUT_DIR/external_multiproperty_eval_rows.summary.json}"
TASK_SPEC_JSON="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TASK_SPEC_JSON:-$OUTPUT_DIR/external_multiproperty_task_specs.json}"
TRAIN_FEATURES_DIR="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TRAIN_FEATURES_DIR:-$OUTPUT_DIR/train_condition_features_hf_vlm}"
EVAL_FEATURES_DIR="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_EVAL_FEATURES_DIR:-$OUTPUT_DIR/eval_condition_features_hf_vlm}"
MODEL_DIR="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_source_edit_sft}"
BENCHMARK_OUTPUT_DIR="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_source_edit_sft}"
PREDICTION_CSV="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_predictions.csv}"
GENERATED_PROPERTIES_CSV="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_GENERATED_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}}"
SOURCE_PROPERTIES_CSV="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_SOURCE_PROPERTIES_CSV:-${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}}"

SUITE="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_SUITE:-${SUCC_EXTERNAL_MULTIPROP_SUITE:-mumo}}"
TASK_SPLIT="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TASK_SPLIT:-${SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT:-all}}"
TASKS="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TASKS:-${SUCC_EXTERNAL_MULTIPROP_TASKS:-}}"
if [[ -n "${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TRAIN_INPUT_SPLIT:-}" ]]; then
  TRAIN_INPUT_SPLIT="$SUCC_EXTERNAL_SOURCE_EDIT_SFT_TRAIN_INPUT_SPLIT"
elif [[ -n "$TRAIN_SOURCE_FILE" && -n "$EVAL_SOURCE_FILE" && "$TRAIN_SOURCE_FILE" != "$EVAL_SOURCE_FILE" ]]; then
  TRAIN_INPUT_SPLIT="all"
else
  TRAIN_INPUT_SPLIT="train"
fi
if [[ -n "${SUCC_EXTERNAL_SOURCE_EDIT_SFT_EVAL_INPUT_SPLIT:-}" ]]; then
  EVAL_INPUT_SPLIT="$SUCC_EXTERNAL_SOURCE_EDIT_SFT_EVAL_INPUT_SPLIT"
elif [[ -n "$TRAIN_SOURCE_FILE" && -n "$EVAL_SOURCE_FILE" && "$TRAIN_SOURCE_FILE" != "$EVAL_SOURCE_FILE" ]]; then
  EVAL_INPUT_SPLIT="all"
else
  EVAL_INPUT_SPLIT="test,eval,valid,validation"
fi
MAX_ROWS_PER_TASK="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_MAX_ROWS_PER_TASK:-${SUCC_EXTERNAL_MULTIPROP_MAX_ROWS_PER_TASK:-200}}"
SEED="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_SEED:-17}"
FORCE_EXPORT="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_FORCE_EXPORT:-${SUCC_EXTERNAL_MULTIPROP_FORCE_EXPORT:-0}}"
RUN_FEATURE_EXPORT="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_RUN_FEATURE_EXPORT:-${SUCC_EXTERNAL_MULTIPROP_RUN_FEATURE_EXPORT:-auto}}"
RESUME_CHECKPOINT="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_RESUME_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
CONDITION_MIXING_MODE="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_CONDITION_MIXING_MODE:-append_source_property_program}"
DISABLE_PROPERTY_RERANK="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_DISABLE_PROPERTY_RERANK:-1}"
RESET_TRAINING_STATE="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_RESET_TRAINING_STATE:-1}"

EPOCHS="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_EPOCHS:-1}"
BATCH_SIZE="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_EVAL_BATCH_SIZE:-32}"
LR="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_LR:-1e-5}"
WEIGHT_DECAY="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_GRAD_CLIP:-1.0}"
MAX_SMILES_LENGTH="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_MAX_SMILES_LENGTH:-160}"
MAX_NEW_TOKENS="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_MAX_NEW_TOKENS:-100}"
TEMPERATURE="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TEMPERATURE:-0.70}"
TOP_K="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TOP_K:-24}"
TOP_P="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_TOP_P:-0.90}"
NUM_SAMPLES="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_NUM_SAMPLES:-20}"
PARALLEL_SAMPLES="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_PARALLEL_SAMPLES:-4}"
MAX_PARALLEL_SEQUENCES="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_MAX_PARALLEL_SEQUENCES:-512}"
REPETITION_PENALTY="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_REPETITION_PENALTY:-1.15}"
NO_REPEAT_NGRAM_SIZE="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_NO_REPEAT_NGRAM_SIZE:-6}"
MIN_NEW_TOKENS="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_MIN_NEW_TOKENS:-6}"
MIN_SOURCE_TANIMOTO="${SUCC_EXTERNAL_SOURCE_EDIT_SFT_MIN_SOURCE_TANIMOTO:-${SUCC_EXTERNAL_MULTIPROP_MIN_SOURCE_TANIMOTO:-0.4}}"
DEVICE="${SUCC_DEVICE:-auto}"

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

mkdir -p "$OUTPUT_DIR" "$MODEL_DIR" "$BENCHMARK_OUTPUT_DIR"

echo "External multi-property source-edit SFT warm-start"
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
echo "  condition_mixing_mode=$CONDITION_MIXING_MODE"
echo "  reset_training_state=$RESET_TRAINING_STATE"
echo "  epochs=$EPOCHS"
echo "  num_samples=$NUM_SAMPLES"

export_rows() {
  local source_file="$1"
  local output_csv="$2"
  local summary_json="$3"
  local task_spec_json="$4"
  local input_split="$5"
  if [[ "$FORCE_EXPORT" == "1" || ! -f "$output_csv" ]]; then
    if [[ -z "$source_file" || ! -f "$source_file" ]]; then
      echo "ERROR: missing external source file for $output_csv." >&2
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

FEATURE_ARGS=()
if [[ -f "$TRAIN_FEATURES_DIR/query_tokens.npy" ]]; then
  FEATURE_ARGS+=(--condition-features-dir "$TRAIN_FEATURES_DIR")
fi
if [[ -f "$EVAL_FEATURES_DIR/query_tokens.npy" ]]; then
  FEATURE_ARGS+=(--eval-condition-features-dir "$EVAL_FEATURES_DIR")
fi

TRAIN_CMD=(
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py"
  --train-csv "$TRAIN_ROWS_CSV"
  --eval-csv "$EVAL_ROWS_CSV"
  --resume-checkpoint "$RESUME_CHECKPOINT"
  "${FEATURE_ARGS[@]}"
  --condition-mixing-mode "$CONDITION_MIXING_MODE"
  --output-dir "$MODEL_DIR"
  --prediction-csv "$PREDICTION_CSV"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --grad-clip "$GRAD_CLIP"
  --max-smiles-length "$MAX_SMILES_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --top-p "$TOP_P"
  --num-samples "$NUM_SAMPLES"
  --parallel-samples "$PARALLEL_SAMPLES"
  --max-parallel-sequences "$MAX_PARALLEL_SEQUENCES"
  --repetition-penalty "$REPETITION_PENALTY"
  --no-repeat-ngram-size "$NO_REPEAT_NGRAM_SIZE"
  --min-new-tokens "$MIN_NEW_TOKENS"
  --seed "$SEED"
  --device "$DEVICE"
)
if [[ "$DISABLE_PROPERTY_RERANK" == "1" ]]; then
  TRAIN_CMD+=(--disable-property-rerank)
fi
if [[ "$RESET_TRAINING_STATE" == "1" ]]; then
  TRAIN_CMD+=(--reset-training-state)
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
  --report-title "SUCC Direct SMILES External Source-edit SFT Benchmark" \
  "${EVAL_ARGS[@]}"

echo
echo "External source-edit SFT benchmark ready:"
echo "  train_rows=$TRAIN_ROWS_CSV"
echo "  eval_rows=$EVAL_ROWS_CSV"
echo "  checkpoint=$MODEL_DIR/direct_smiles_generator.pt"
echo "  predictions=$PREDICTION_CSV"
echo "  report=$BENCHMARK_OUTPUT_DIR/external_multiproperty_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/external_multiproperty_summary.csv"
