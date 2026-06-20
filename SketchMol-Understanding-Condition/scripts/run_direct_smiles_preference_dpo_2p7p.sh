#!/usr/bin/env bash
# Build direct-SMILES preference pairs and run a DPO-style 2p-7p fine-tune.

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
BASE_OUTPUT_DIR="${SUCC_DIRECT_DPO_BASE_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v1_sampled_rerank}"
OUTPUT_DIR="${SUCC_DIRECT_DPO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v1_dpo_v1}"
TRAIN_ROWS_CSV="${SUCC_DIRECT_DPO_TRAIN_ROWS_CSV:-$BASE_OUTPUT_DIR/denovo_2p7p_train_rows.csv}"
EVAL_ROWS_CSV="${SUCC_DIRECT_DPO_EVAL_ROWS_CSV:-$BASE_OUTPUT_DIR/denovo_2p7p_eval_rows.csv}"
TRAIN_FEATURES_DIR="${SUCC_DIRECT_DPO_TRAIN_FEATURES_DIR:-$BASE_OUTPUT_DIR/train_condition_features_hf_vlm}"
EVAL_FEATURES_DIR="${SUCC_DIRECT_DPO_EVAL_FEATURES_DIR:-$BASE_OUTPUT_DIR/eval_condition_features_hf_vlm}"
RESUME_CHECKPOINT="${SUCC_DIRECT_DPO_RESUME_CHECKPOINT:-$BASE_OUTPUT_DIR/direct_smiles_model/direct_smiles_generator.pt}"
TRAIN_PREFERENCE_CSV="${SUCC_DIRECT_DPO_TRAIN_PREFERENCE_CSV:-$OUTPUT_DIR/train_preferences.csv}"
EVAL_PREFERENCE_CSV="${SUCC_DIRECT_DPO_EVAL_PREFERENCE_CSV:-$OUTPUT_DIR/eval_preferences.csv}"
MODEL_DIR="${SUCC_DIRECT_DPO_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_dpo}"
RUN_BUILD_PREFERENCES="${SUCC_DIRECT_DPO_RUN_BUILD_PREFERENCES:-1}"
FORCE_BUILD_PREFERENCES="${SUCC_DIRECT_DPO_FORCE_BUILD_PREFERENCES:-0}"
RUN_BENCHMARK_AFTER_TRAIN="${SUCC_DIRECT_DPO_RUN_BENCHMARK_AFTER_TRAIN:-1}"
BENCHMARK_OUTPUT_DIR="${SUCC_DIRECT_DPO_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_direct_smiles_dpo}"
BENCHMARK_MODEL_DIR="${SUCC_DIRECT_DPO_BENCHMARK_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_dpo_eval}"
BENCHMARK_PREDICTION_CSV="${SUCC_DIRECT_DPO_BENCHMARK_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_predictions.csv}"

PREF_BATCH_SIZE="${SUCC_DIRECT_DPO_PREF_BATCH_SIZE:-128}"
PREF_NUM_SAMPLES="${SUCC_DIRECT_DPO_PREF_NUM_SAMPLES:-16}"
PREF_PARALLEL_SAMPLES="${SUCC_DIRECT_DPO_PREF_PARALLEL_SAMPLES:-8}"
PREF_MAX_PARALLEL_SEQUENCES="${SUCC_DIRECT_DPO_PREF_MAX_PARALLEL_SEQUENCES:-1024}"
PREF_MAX_NEW_TOKENS="${SUCC_DIRECT_DPO_PREF_MAX_NEW_TOKENS:-96}"
PREF_TEMPERATURE="${SUCC_DIRECT_DPO_PREF_TEMPERATURE:-0.85}"
PREF_TOP_K="${SUCC_DIRECT_DPO_PREF_TOP_K:-40}"
PREF_TOP_P="${SUCC_DIRECT_DPO_PREF_TOP_P:-0.95}"
PREF_REPETITION_PENALTY="${SUCC_DIRECT_DPO_PREF_REPETITION_PENALTY:-1.15}"
PREF_NO_REPEAT_NGRAM_SIZE="${SUCC_DIRECT_DPO_PREF_NO_REPEAT_NGRAM_SIZE:-6}"
PREF_MIN_NEW_TOKENS="${SUCC_DIRECT_DPO_PREF_MIN_NEW_TOKENS:-6}"
PREF_MIN_SCORE_GAP="${SUCC_DIRECT_DPO_PREF_MIN_SCORE_GAP:-0.5}"
PREF_REJECTED_STRATEGY="${SUCC_DIRECT_DPO_PREF_REJECTED_STRATEGY:-hard_valid}"
PREF_TRAIN_LIMIT="${SUCC_DIRECT_DPO_PREF_TRAIN_LIMIT:-0}"
PREF_EVAL_LIMIT="${SUCC_DIRECT_DPO_PREF_EVAL_LIMIT:-0}"

DPO_EPOCHS="${SUCC_DIRECT_DPO_EPOCHS:-1}"
DPO_BATCH_SIZE="${SUCC_DIRECT_DPO_BATCH_SIZE:-32}"
DPO_EVAL_BATCH_SIZE="${SUCC_DIRECT_DPO_EVAL_BATCH_SIZE:-64}"
DPO_LR="${SUCC_DIRECT_DPO_LR:-5e-6}"
DPO_WEIGHT_DECAY="${SUCC_DIRECT_DPO_WEIGHT_DECAY:-1e-4}"
DPO_GRAD_CLIP="${SUCC_DIRECT_DPO_GRAD_CLIP:-1.0}"
DPO_BETA="${SUCC_DIRECT_DPO_BETA:-0.1}"
DPO_SFT_WEIGHT="${SUCC_DIRECT_DPO_SFT_WEIGHT:-0.5}"
DEVICE="${SUCC_DEVICE:-auto}"
SEED="${SUCC_DIRECT_DPO_SEED:-7}"

BENCHMARK_MAX_NEW_TOKENS="${SUCC_DIRECT_DPO_BENCHMARK_MAX_NEW_TOKENS:-96}"
BENCHMARK_TEMPERATURE="${SUCC_DIRECT_DPO_BENCHMARK_TEMPERATURE:-0.85}"
BENCHMARK_TOP_K="${SUCC_DIRECT_DPO_BENCHMARK_TOP_K:-40}"
BENCHMARK_TOP_P="${SUCC_DIRECT_DPO_BENCHMARK_TOP_P:-0.95}"
BENCHMARK_NUM_SAMPLES="${SUCC_DIRECT_DPO_BENCHMARK_NUM_SAMPLES:-256}"
BENCHMARK_PARALLEL_SAMPLES="${SUCC_DIRECT_DPO_BENCHMARK_PARALLEL_SAMPLES:-8}"
BENCHMARK_MAX_PARALLEL_SEQUENCES="${SUCC_DIRECT_DPO_BENCHMARK_MAX_PARALLEL_SEQUENCES:-1024}"
BENCHMARK_REPETITION_PENALTY="${SUCC_DIRECT_DPO_BENCHMARK_REPETITION_PENALTY:-1.15}"
BENCHMARK_NO_REPEAT_NGRAM_SIZE="${SUCC_DIRECT_DPO_BENCHMARK_NO_REPEAT_NGRAM_SIZE:-6}"
BENCHMARK_MIN_NEW_TOKENS="${SUCC_DIRECT_DPO_BENCHMARK_MIN_NEW_TOKENS:-6}"

mkdir -p "$OUTPUT_DIR" "$MODEL_DIR" "$BENCHMARK_OUTPUT_DIR" "$BENCHMARK_MODEL_DIR"

echo "Direct-SMILES preference DPO (2p-7p)"
echo "  python=$PYTHON_BIN"
echo "  base_output_dir=$BASE_OUTPUT_DIR"
echo "  output_dir=$OUTPUT_DIR"
echo "  resume_checkpoint=$RESUME_CHECKPOINT"
echo "  train_preference_csv=$TRAIN_PREFERENCE_CSV"
echo "  eval_preference_csv=$EVAL_PREFERENCE_CSV"
echo "  pref_num_samples=$PREF_NUM_SAMPLES"
echo "  pref_rejected_strategy=$PREF_REJECTED_STRATEGY"
echo "  dpo_epochs=$DPO_EPOCHS"
echo "  dpo_lr=$DPO_LR"
echo "  run_benchmark_after_train=$RUN_BENCHMARK_AFTER_TRAIN"
echo "  benchmark_output_dir=$BENCHMARK_OUTPUT_DIR"
echo "  benchmark_num_samples=$BENCHMARK_NUM_SAMPLES"

build_preferences() {
  local rows_csv="$1"
  local features_dir="$2"
  local output_csv="$3"
  local limit="$4"
  local should_build=0
  if [[ "$FORCE_BUILD_PREFERENCES" == "1" || ! -f "$output_csv" ]]; then
    should_build=1
  fi
  if [[ "$should_build" != "1" ]]; then
    return
  fi
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/build_direct_smiles_preference_dataset.py" \
    --rows-csv "$rows_csv" \
    --output-csv "$output_csv" \
    --resume-checkpoint "$RESUME_CHECKPOINT" \
    --condition-features-dir "$features_dir" \
    --limit "$limit" \
    --batch-size "$PREF_BATCH_SIZE" \
    --num-samples "$PREF_NUM_SAMPLES" \
    --parallel-samples "$PREF_PARALLEL_SAMPLES" \
    --max-parallel-sequences "$PREF_MAX_PARALLEL_SEQUENCES" \
    --max-new-tokens "$PREF_MAX_NEW_TOKENS" \
    --temperature "$PREF_TEMPERATURE" \
    --top-k "$PREF_TOP_K" \
    --top-p "$PREF_TOP_P" \
    --repetition-penalty "$PREF_REPETITION_PENALTY" \
    --no-repeat-ngram-size "$PREF_NO_REPEAT_NGRAM_SIZE" \
    --min-new-tokens "$PREF_MIN_NEW_TOKENS" \
    --rejected-strategy "$PREF_REJECTED_STRATEGY" \
    --min-score-gap "$PREF_MIN_SCORE_GAP" \
    --seed "$SEED" \
    --device "$DEVICE"
}

if [[ "$RUN_BUILD_PREFERENCES" == "1" ]]; then
  build_preferences "$TRAIN_ROWS_CSV" "$TRAIN_FEATURES_DIR" "$TRAIN_PREFERENCE_CSV" "$PREF_TRAIN_LIMIT"
  build_preferences "$EVAL_ROWS_CSV" "$EVAL_FEATURES_DIR" "$EVAL_PREFERENCE_CSV" "$PREF_EVAL_LIMIT"
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator_dpo.py" \
  --train-preference-csv "$TRAIN_PREFERENCE_CSV" \
  --eval-preference-csv "$EVAL_PREFERENCE_CSV" \
  --output-dir "$MODEL_DIR" \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  --condition-features-dir "$TRAIN_FEATURES_DIR" \
  --eval-condition-features-dir "$EVAL_FEATURES_DIR" \
  --epochs "$DPO_EPOCHS" \
  --batch-size "$DPO_BATCH_SIZE" \
  --eval-batch-size "$DPO_EVAL_BATCH_SIZE" \
  --lr "$DPO_LR" \
  --weight-decay "$DPO_WEIGHT_DECAY" \
  --grad-clip "$DPO_GRAD_CLIP" \
  --beta "$DPO_BETA" \
  --sft-weight "$DPO_SFT_WEIGHT" \
  --seed "$SEED" \
  --device "$DEVICE"

DPO_CHECKPOINT="$MODEL_DIR/direct_smiles_generator_dpo.pt"
if [[ "$RUN_BENCHMARK_AFTER_TRAIN" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py" \
    --eval-only \
    --eval-csv "$EVAL_ROWS_CSV" \
    --resume-checkpoint "$DPO_CHECKPOINT" \
    --condition-features-dir "$TRAIN_FEATURES_DIR" \
    --eval-condition-features-dir "$EVAL_FEATURES_DIR" \
    --output-dir "$BENCHMARK_MODEL_DIR" \
    --prediction-csv "$BENCHMARK_PREDICTION_CSV" \
    --eval-batch-size "$DPO_EVAL_BATCH_SIZE" \
    --max-new-tokens "$BENCHMARK_MAX_NEW_TOKENS" \
    --temperature "$BENCHMARK_TEMPERATURE" \
    --top-k "$BENCHMARK_TOP_K" \
    --top-p "$BENCHMARK_TOP_P" \
    --num-samples "$BENCHMARK_NUM_SAMPLES" \
    --parallel-samples "$BENCHMARK_PARALLEL_SAMPLES" \
    --max-parallel-sequences "$BENCHMARK_MAX_PARALLEL_SEQUENCES" \
    --repetition-penalty "$BENCHMARK_REPETITION_PENALTY" \
    --no-repeat-ngram-size "$BENCHMARK_NO_REPEAT_NGRAM_SIZE" \
    --min-new-tokens "$BENCHMARK_MIN_NEW_TOKENS" \
    --seed "$SEED" \
    --device "$DEVICE"

  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_univideo_image_benchmark.py" \
    --image-csv "$BENCHMARK_PREDICTION_CSV" \
    --output-dir "$BENCHMARK_OUTPUT_DIR" \
    --method direct_smiles_mllm \
    --smiles-column generated_smiles \
    --report-title "SUCC Direct SMILES Preference-DPO 2p-7p Benchmark" \
    --benchmark-family "direct_smiles_denovo_property_design" \
    --benchmark-task "direct_smiles_denovo_2p7p_property_design" \
    --accept-direct-smiles \
    --hide-source-similarity-section

  echo
  echo "Direct-SMILES preference-DPO benchmark ready:"
  echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
  echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
  echo "  prediction_csv=$BENCHMARK_PREDICTION_CSV"
  sed -n '1,100p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
fi
