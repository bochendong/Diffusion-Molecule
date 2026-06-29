#!/usr/bin/env bash
# Run SUCC direct-SMILES on external source-conditioned multi-property tasks.

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
OUTPUT_DIR="${SUCC_EXTERNAL_MULTIPROP_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_external_multiproperty_pilot}"
SOURCE_FILE="${SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE:-}"
ROWS_CSV="${SUCC_EXTERNAL_MULTIPROP_ROWS_CSV:-$OUTPUT_DIR/external_multiproperty_rows.csv}"
TASK_SPEC_JSON="${SUCC_EXTERNAL_MULTIPROP_TASK_SPEC_JSON:-$OUTPUT_DIR/external_multiproperty_task_specs.json}"
SUMMARY_JSON="${SUCC_EXTERNAL_MULTIPROP_SUMMARY_JSON:-$OUTPUT_DIR/external_multiproperty_rows.summary.json}"
FEATURES_DIR="${SUCC_EXTERNAL_MULTIPROP_FEATURES_DIR:-$OUTPUT_DIR/condition_features_hf_vlm}"
MODEL_DIR="${SUCC_EXTERNAL_MULTIPROP_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_external}"
BENCHMARK_OUTPUT_DIR="${SUCC_EXTERNAL_MULTIPROP_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_external_multiproperty}"
PREDICTION_CSV="${SUCC_EXTERNAL_MULTIPROP_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_predictions.csv}"
CANDIDATE_PREDICTION_CSV="${SUCC_EXTERNAL_MULTIPROP_CANDIDATE_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_candidate_predictions.csv}"
EVAL_PREDICTION_CSV="${SUCC_EXTERNAL_MULTIPROP_EVAL_PREDICTION_CSV:-$CANDIDATE_PREDICTION_CSV}"
GENERATED_PROPERTIES_CSV="${SUCC_EXTERNAL_MULTIPROP_GENERATED_PROPERTIES_CSV:-}"
SOURCE_PROPERTIES_CSV="${SUCC_EXTERNAL_MULTIPROP_SOURCE_PROPERTIES_CSV:-}"

SUITE="${SUCC_EXTERNAL_MULTIPROP_SUITE:-both}"
TASK_SPLIT="${SUCC_EXTERNAL_MULTIPROP_TASK_SPLIT:-all}"
TASKS="${SUCC_EXTERNAL_MULTIPROP_TASKS:-}"
INPUT_SPLIT="${SUCC_EXTERNAL_MULTIPROP_INPUT_SPLIT:-all}"
MAX_ROWS_PER_TASK="${SUCC_EXTERNAL_MULTIPROP_MAX_ROWS_PER_TASK:-200}"
SEED="${SUCC_EXTERNAL_MULTIPROP_SEED:-17}"
FORCE_EXPORT="${SUCC_EXTERNAL_MULTIPROP_FORCE_EXPORT:-0}"
RUN_FEATURE_EXPORT="${SUCC_EXTERNAL_MULTIPROP_RUN_FEATURE_EXPORT:-auto}"
RUN_TRAIN="${SUCC_EXTERNAL_MULTIPROP_RUN_TRAIN:-0}"
RESUME_CHECKPOINT="${SUCC_EXTERNAL_MULTIPROP_RESUME_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/direct_smiles_model/direct_smiles_generator.pt}"
CONDITION_MIXING_MODE="${SUCC_EXTERNAL_MULTIPROP_CONDITION_MIXING_MODE:-append_property_program}"
DISABLE_PROPERTY_RERANK="${SUCC_EXTERNAL_MULTIPROP_DISABLE_PROPERTY_RERANK:-1}"

EPOCHS="${SUCC_EXTERNAL_MULTIPROP_EPOCHS:-2}"
BATCH_SIZE="${SUCC_EXTERNAL_MULTIPROP_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${SUCC_EXTERNAL_MULTIPROP_EVAL_BATCH_SIZE:-64}"
LR="${SUCC_EXTERNAL_MULTIPROP_LR:-1e-5}"
MAX_SMILES_LENGTH="${SUCC_EXTERNAL_MULTIPROP_MAX_SMILES_LENGTH:-160}"
MAX_NEW_TOKENS="${SUCC_EXTERNAL_MULTIPROP_MAX_NEW_TOKENS:-100}"
TEMPERATURE="${SUCC_EXTERNAL_MULTIPROP_TEMPERATURE:-0.85}"
TOP_K="${SUCC_EXTERNAL_MULTIPROP_TOP_K:-40}"
TOP_P="${SUCC_EXTERNAL_MULTIPROP_TOP_P:-0.95}"
NUM_SAMPLES="${SUCC_EXTERNAL_MULTIPROP_NUM_SAMPLES:-20}"
PARALLEL_SAMPLES="${SUCC_EXTERNAL_MULTIPROP_PARALLEL_SAMPLES:-4}"
MAX_PARALLEL_SEQUENCES="${SUCC_EXTERNAL_MULTIPROP_MAX_PARALLEL_SEQUENCES:-512}"
REPETITION_PENALTY="${SUCC_EXTERNAL_MULTIPROP_REPETITION_PENALTY:-1.15}"
NO_REPEAT_NGRAM_SIZE="${SUCC_EXTERNAL_MULTIPROP_NO_REPEAT_NGRAM_SIZE:-6}"
MIN_NEW_TOKENS="${SUCC_EXTERNAL_MULTIPROP_MIN_NEW_TOKENS:-6}"
MIN_SOURCE_TANIMOTO="${SUCC_EXTERNAL_MULTIPROP_MIN_SOURCE_TANIMOTO:-0.4}"
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

echo "External multi-property direct-SMILES benchmark"
echo "  python=$PYTHON_BIN"
echo "  source_file=${SOURCE_FILE:-none}"
echo "  output_dir=$OUTPUT_DIR"
echo "  rows_csv=$ROWS_CSV"
echo "  suite=$SUITE"
echo "  task_split=$TASK_SPLIT"
echo "  tasks=${TASKS:-all}"
echo "  input_split=$INPUT_SPLIT"
echo "  max_rows_per_task=$MAX_ROWS_PER_TASK"
echo "  resume_checkpoint=$RESUME_CHECKPOINT"
echo "  run_train=$RUN_TRAIN"
echo "  run_feature_export=$RUN_FEATURE_EXPORT"
echo "  condition_mixing_mode=$CONDITION_MIXING_MODE"
echo "  disable_property_rerank=$DISABLE_PROPERTY_RERANK"
echo "  num_samples=$NUM_SAMPLES"

if [[ "$FORCE_EXPORT" == "1" || ! -f "$ROWS_CSV" ]]; then
  if [[ -z "$SOURCE_FILE" || ! -f "$SOURCE_FILE" ]]; then
    echo "ERROR: set SUCC_EXTERNAL_MULTIPROP_SOURCE_FILE to a MuMO/C-MuMO JSON/JSONL/CSV source file." >&2
    exit 2
  fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_external_multiproperty_benchmark_rows.py" \
    --source-file "$SOURCE_FILE" \
    --output-csv "$ROWS_CSV" \
    --summary-json "$SUMMARY_JSON" \
    --task-spec-json "$TASK_SPEC_JSON" \
    --suite "$SUITE" \
    --task-split "$TASK_SPLIT" \
    --tasks "$TASKS" \
    --input-split "$INPUT_SPLIT" \
    --max-rows-per-task "$MAX_ROWS_PER_TASK" \
    --seed "$SEED"
fi

should_export_features=0
if [[ "$RUN_FEATURE_EXPORT" == "1" || "$FORCE_EXPORT" == "1" ]]; then
  should_export_features=1
elif [[ "$RUN_FEATURE_EXPORT" == "auto" && ! -f "$FEATURES_DIR/query_tokens.npy" ]]; then
  should_export_features=1
fi
if [[ "$should_export_features" == "1" ]]; then
  export SUCC_ENCODER=hf_vlm
  export SUCC_VARIANTS=full
  export SUCC_BASELINE_CSV="$ROWS_CSV"
  export SUCC_OUTPUT_DIR="$FEATURES_DIR"
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

TRAIN_ARGS=()
if [[ "$RUN_TRAIN" == "1" ]]; then
  TRAIN_ARGS+=(--train-csv "$ROWS_CSV")
else
  TRAIN_ARGS+=(--eval-only)
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

FEATURE_ARGS=()
if [[ -f "$FEATURES_DIR/query_tokens.npy" ]]; then
  FEATURE_ARGS+=(--condition-features-dir "$FEATURES_DIR" --eval-condition-features-dir "$FEATURES_DIR")
fi

TRAIN_CMD=(
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py"
  "${TRAIN_ARGS[@]}"
  --eval-csv "$ROWS_CSV"
  "${FEATURE_ARGS[@]}"
  --condition-mixing-mode "$CONDITION_MIXING_MODE"
  --output-dir "$MODEL_DIR"
  --prediction-csv "$PREDICTION_CSV"
  --candidate-output-csv "$CANDIDATE_PREDICTION_CSV"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --lr "$LR"
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
"${TRAIN_CMD[@]}"

EVAL_ARGS=()
if [[ -n "$GENERATED_PROPERTIES_CSV" ]]; then
  EVAL_ARGS+=(--generated-properties-csv "$GENERATED_PROPERTIES_CSV")
fi
if [[ -n "$SOURCE_PROPERTIES_CSV" ]]; then
  EVAL_ARGS+=(--source-properties-csv "$SOURCE_PROPERTIES_CSV")
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
  --prediction-csv "$EVAL_PREDICTION_CSV" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --smiles-column generated_smiles \
  --source-smiles-column source_smiles \
  --group-column condition_id \
  --min-source-tanimoto "$MIN_SOURCE_TANIMOTO" \
  --report-title "SUCC Direct SMILES External Multi-property Benchmark" \
  "${EVAL_ARGS[@]}"

echo
echo "External multi-property benchmark ready:"
echo "  rows=$ROWS_CSV"
echo "  task_specs=$TASK_SPEC_JSON"
echo "  predictions=$PREDICTION_CSV"
echo "  candidate_predictions=$CANDIDATE_PREDICTION_CSV"
echo "  evaluated_predictions=$EVAL_PREDICTION_CSV"
echo "  report=$BENCHMARK_OUTPUT_DIR/external_multiproperty_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/external_multiproperty_summary.csv"
