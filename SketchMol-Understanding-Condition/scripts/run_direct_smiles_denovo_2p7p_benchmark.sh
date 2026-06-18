#!/usr/bin/env bash
# Train/evaluate the MLLM-conditioned direct SMILES generator on de novo 2p-7p.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

# shellcheck source=./multiproperty_dataset_defaults.sh
source "$SCRIPT_DIR/multiproperty_dataset_defaults.sh"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
OUTPUT_DIR="${SUCC_DIRECT_DENOVO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank}"
MOLECULE_DB="${SUCC_DIRECT_DENOVO_MOLECULE_DB_CSV:-$SMMED_DEFAULT_MOLECULE_DB}"
EVAL_ROWS_CSV="${SUCC_DIRECT_DENOVO_EVAL_ROWS_CSV:-$OUTPUT_DIR/denovo_2p7p_eval_rows.csv}"
TRAIN_ROWS_CSV="${SUCC_DIRECT_DENOVO_TRAIN_ROWS_CSV:-$OUTPUT_DIR/denovo_2p7p_train_rows.csv}"
CANDIDATE_ROWS_CSV="${SUCC_DIRECT_DENOVO_CANDIDATE_ROWS_CSV:-$OUTPUT_DIR/denovo_2p7p_candidate_rows.csv}"
TRAIN_FEATURES_DIR="${SUCC_DIRECT_DENOVO_TRAIN_FEATURES_DIR:-$OUTPUT_DIR/train_condition_features_hf_vlm}"
EVAL_FEATURES_DIR="${SUCC_DIRECT_DENOVO_EVAL_FEATURES_DIR:-$OUTPUT_DIR/eval_condition_features_hf_vlm}"
MODEL_DIR="${SUCC_DIRECT_DENOVO_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model}"
BENCHMARK_OUTPUT_DIR="${SUCC_DIRECT_DENOVO_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_direct_smiles}"
PREDICTION_CSV="${SUCC_DIRECT_DENOVO_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_predictions.csv}"
ROWS_PER_PROPERTY_COUNT="${SUCC_DIRECT_DENOVO_EVAL_ROWS_PER_PROPERTY_COUNT:-1000}"
TRAIN_ROWS_PER_PROPERTY_COUNT="${SUCC_DIRECT_DENOVO_TRAIN_ROWS_PER_PROPERTY_COUNT:-2000}"
MIN_PROPERTIES="${SUCC_DIRECT_DENOVO_MIN_PROPERTIES:-2}"
MAX_PROPERTIES="${SUCC_DIRECT_DENOVO_MAX_PROPERTIES:-7}"
SEED="${SUCC_DIRECT_DENOVO_SEED:-31}"
FORCE_EXPORT="${SUCC_DIRECT_DENOVO_FORCE_EXPORT:-0}"
RUN_FEATURE_EXPORT="${SUCC_DIRECT_DENOVO_RUN_FEATURE_EXPORT:-auto}"
RUN_TRAIN="${SUCC_DIRECT_DENOVO_RUN_TRAIN:-1}"
RESUME_CHECKPOINT="${SUCC_DIRECT_DENOVO_RESUME_CHECKPOINT:-}"
EPOCHS="${SUCC_DIRECT_DENOVO_EPOCHS:-12}"
BATCH_SIZE="${SUCC_DIRECT_DENOVO_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${SUCC_DIRECT_DENOVO_EVAL_BATCH_SIZE:-128}"
LR="${SUCC_DIRECT_DENOVO_LR:-3e-4}"
D_MODEL="${SUCC_DIRECT_DENOVO_D_MODEL:-256}"
NUM_LAYERS="${SUCC_DIRECT_DENOVO_NUM_LAYERS:-4}"
NUM_HEADS="${SUCC_DIRECT_DENOVO_NUM_HEADS:-8}"
MAX_SMILES_LENGTH="${SUCC_DIRECT_DENOVO_MAX_SMILES_LENGTH:-160}"
MAX_NEW_TOKENS="${SUCC_DIRECT_DENOVO_MAX_NEW_TOKENS:-96}"
TEMPERATURE="${SUCC_DIRECT_DENOVO_TEMPERATURE:-0.85}"
TOP_K="${SUCC_DIRECT_DENOVO_TOP_K:-40}"
TOP_P="${SUCC_DIRECT_DENOVO_TOP_P:-0.95}"
NUM_SAMPLES="${SUCC_DIRECT_DENOVO_NUM_SAMPLES:-32}"
PARALLEL_SAMPLES="${SUCC_DIRECT_DENOVO_PARALLEL_SAMPLES:-8}"
MAX_PARALLEL_SEQUENCES="${SUCC_DIRECT_DENOVO_MAX_PARALLEL_SEQUENCES:-1024}"
REPETITION_PENALTY="${SUCC_DIRECT_DENOVO_REPETITION_PENALTY:-1.15}"
NO_REPEAT_NGRAM_SIZE="${SUCC_DIRECT_DENOVO_NO_REPEAT_NGRAM_SIZE:-6}"
MIN_NEW_TOKENS="${SUCC_DIRECT_DENOVO_MIN_NEW_TOKENS:-6}"
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

echo "Direct SMILES de novo 2p-7p benchmark"
echo "  python=$PYTHON_BIN"
echo "  molecule_db=$MOLECULE_DB"
echo "  output_dir=$OUTPUT_DIR"
echo "  train_rows=$TRAIN_ROWS_CSV"
echo "  eval_rows=$EVAL_ROWS_CSV"
echo "  model_dir=$MODEL_DIR"
echo "  prediction_csv=$PREDICTION_CSV"
echo "  train_rows_per_property_count=$TRAIN_ROWS_PER_PROPERTY_COUNT"
echo "  eval_rows_per_property_count=$ROWS_PER_PROPERTY_COUNT"
echo "  num_samples=$NUM_SAMPLES"
echo "  parallel_samples=$PARALLEL_SAMPLES"
echo "  max_parallel_sequences=$MAX_PARALLEL_SEQUENCES"
echo "  decoding=temperature:$TEMPERATURE top_k:$TOP_K top_p:$TOP_P repetition_penalty:$REPETITION_PENALTY no_repeat_ngram:$NO_REPEAT_NGRAM_SIZE"

if [[ ! -f "$MOLECULE_DB" ]]; then
  echo "ERROR: missing molecule database: $MOLECULE_DB" >&2
  exit 2
fi

if [[ "$FORCE_EXPORT" == "1" || ! -f "$EVAL_ROWS_CSV" || ! -f "$TRAIN_ROWS_CSV" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/export_denovo_2p7p_benchmark_rows.py" \
    --molecule-db-csv "$MOLECULE_DB" \
    --output-csv "$EVAL_ROWS_CSV" \
    --candidate-output-csv "$CANDIDATE_ROWS_CSV" \
    --train-output-csv "$TRAIN_ROWS_CSV" \
    --rows-per-property-count "$ROWS_PER_PROPERTY_COUNT" \
    --train-rows-per-property-count "$TRAIN_ROWS_PER_PROPERTY_COUNT" \
    --min-properties "$MIN_PROPERTIES" \
    --max-properties "$MAX_PROPERTIES" \
    --seed "$SEED"
fi

export_features() {
  local csv_path="$1"
  local features_dir="$2"
  local should_export=0
  if [[ "$RUN_FEATURE_EXPORT" == "1" || "$FORCE_EXPORT" == "1" ]]; then
    should_export=1
  elif [[ "$RUN_FEATURE_EXPORT" == "auto" && ! -f "$features_dir/query_tokens.npy" ]]; then
    should_export=1
  fi
  if [[ "$should_export" != "1" ]]; then
    return
  fi
  export SUCC_ENCODER=hf_vlm
  export SUCC_VARIANTS=full
  export SUCC_BASELINE_CSV="$csv_path"
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
}

export_features "$TRAIN_ROWS_CSV" "$TRAIN_FEATURES_DIR"
export_features "$EVAL_ROWS_CSV" "$EVAL_FEATURES_DIR"

TRAIN_ARGS=()
if [[ "$RUN_TRAIN" == "1" ]]; then
  TRAIN_ARGS+=(--train-csv "$TRAIN_ROWS_CSV")
else
  TRAIN_ARGS+=(--eval-only)
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py" \
  "${TRAIN_ARGS[@]}" \
  --eval-csv "$EVAL_ROWS_CSV" \
  --condition-features-dir "$TRAIN_FEATURES_DIR" \
  --eval-condition-features-dir "$EVAL_FEATURES_DIR" \
  --output-dir "$MODEL_DIR" \
  --prediction-csv "$PREDICTION_CSV" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --lr "$LR" \
  --d-model "$D_MODEL" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --max-smiles-length "$MAX_SMILES_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --top-p "$TOP_P" \
  --num-samples "$NUM_SAMPLES" \
  --parallel-samples "$PARALLEL_SAMPLES" \
  --max-parallel-sequences "$MAX_PARALLEL_SEQUENCES" \
  --repetition-penalty "$REPETITION_PENALTY" \
  --no-repeat-ngram-size "$NO_REPEAT_NGRAM_SIZE" \
  --min-new-tokens "$MIN_NEW_TOKENS" \
  --seed "$SEED" \
  --device "$DEVICE"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_univideo_image_benchmark.py" \
  --image-csv "$PREDICTION_CSV" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --method direct_smiles_mllm \
  --smiles-column generated_smiles \
  --report-title "SUCC Direct SMILES De Novo 2p-7p Benchmark" \
  --benchmark-family "direct_smiles_denovo_property_design" \
  --benchmark-task "direct_smiles_denovo_2p7p_property_design" \
  --accept-direct-smiles \
  --hide-source-similarity-section

echo
echo "Direct SMILES de novo 2p-7p benchmark ready:"
echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
echo "  prediction_csv=$PREDICTION_CSV"
sed -n '1,100p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
