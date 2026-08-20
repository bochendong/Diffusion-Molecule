#!/usr/bin/env bash
# Fast GSK3B n=20 one-shot pilot: reward the actual assay + source similarity, no ranking.

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
SOURCE_ROOT="${SUCC_TABLE1_GSK3B_SOURCE_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_moledit_table1_group_rl_v1}"
OUTPUT_DIR="${SUCC_TABLE1_GSK3B_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_moledit_table1_gsk3b_n20_pilot_v1}"
PACK_DIR="${SUCC_TABLE1_GSK3B_PACK_DIR:-$OUTPUT_DIR/gsk3b_pack}"
TRAIN_FEATURES_DIR="${SUCC_TABLE1_GSK3B_TRAIN_FEATURES_DIR:-$SOURCE_ROOT/train_condition_features_hf_vlm}"
EVAL_FEATURES_DIR="${SUCC_TABLE1_GSK3B_EVAL_FEATURES_DIR:-$OUTPUT_DIR/eval_condition_features_hf_vlm}"
SFT_CHECKPOINT="${SUCC_TABLE1_GSK3B_SFT_CHECKPOINT:-$SOURCE_ROOT/direct_smiles_model_source_edit_sft/direct_smiles_generator.pt}"
RL_MODEL_DIR="${SUCC_TABLE1_GSK3B_RL_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_group_rl}"
BENCHMARK_MODEL_DIR="${SUCC_TABLE1_GSK3B_BENCHMARK_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_group_rl_eval}"
BENCHMARK_OUTPUT_DIR="${SUCC_TABLE1_GSK3B_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_n20}"
CANDIDATE_PREDICTION_CSV="${SUCC_TABLE1_GSK3B_CANDIDATE_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_candidate_predictions.csv}"
RAW_SELECTED_CSV="${SUCC_TABLE1_GSK3B_RAW_SELECTED_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_selected_raw.csv}"
RL_CHECKPOINT="${SUCC_TABLE1_GSK3B_RL_CHECKPOINT:-$RL_MODEL_DIR/direct_smiles_generator_rl.pt}"

EVAL_LIMIT="${SUCC_TABLE1_GSK3B_EVAL_LIMIT:-40}"
CONDITION_MIXING_MODE="${SUCC_TABLE1_GSK3B_CONDITION_MIXING_MODE:-append_source_property_program}"
DEVICE="${SUCC_DEVICE:-auto}"
SEED="${SUCC_TABLE1_GSK3B_SEED:-23}"

RL_EPOCHS="${SUCC_TABLE1_GSK3B_RL_EPOCHS:-1}"
RL_BATCH_SIZE="${SUCC_TABLE1_GSK3B_RL_BATCH_SIZE:-8}"
RL_EVAL_BATCH_SIZE="${SUCC_TABLE1_GSK3B_RL_EVAL_BATCH_SIZE:-16}"
RL_LR="${SUCC_TABLE1_GSK3B_RL_LR:-5e-7}"
RL_ROLLOUTS_PER_PROMPT="${SUCC_TABLE1_GSK3B_ROLLOUTS_PER_PROMPT:-8}"
RL_PARALLEL_SAMPLES="${SUCC_TABLE1_GSK3B_PARALLEL_SAMPLES:-8}"
RL_MAX_PARALLEL_SEQUENCES="${SUCC_TABLE1_GSK3B_MAX_PARALLEL_SEQUENCES:-512}"
RL_MAX_NEW_TOKENS="${SUCC_TABLE1_GSK3B_MAX_NEW_TOKENS:-100}"
RL_TEMPERATURE="${SUCC_TABLE1_GSK3B_TEMPERATURE:-0.85}"
RL_TOP_K="${SUCC_TABLE1_GSK3B_TOP_K:-40}"
RL_TOP_P="${SUCC_TABLE1_GSK3B_TOP_P:-0.95}"
RL_REPETITION_PENALTY="${SUCC_TABLE1_GSK3B_REPETITION_PENALTY:-1.15}"
RL_NO_REPEAT_NGRAM_SIZE="${SUCC_TABLE1_GSK3B_NO_REPEAT_NGRAM_SIZE:-6}"
RL_MIN_NEW_TOKENS="${SUCC_TABLE1_GSK3B_MIN_NEW_TOKENS:-6}"
RL_SFT_WEIGHT="${SUCC_TABLE1_GSK3B_SFT_WEIGHT:-1.0}"
RL_ADVANTAGE_MODE="${SUCC_TABLE1_GSK3B_ADVANTAGE_MODE:-group_zscore}"
RL_REFERENCE_KL_WEIGHT="${SUCC_TABLE1_GSK3B_REFERENCE_KL_WEIGHT:-0.05}"
RL_REWARD_VALID_WEIGHT="${SUCC_TABLE1_GSK3B_REWARD_VALID_WEIGHT:-0.25}"
RL_REWARD_STRICT_WEIGHT="${SUCC_TABLE1_GSK3B_REWARD_STRICT_WEIGHT:-3.0}"
RL_REWARD_DISTANCE_WEIGHT="${SUCC_TABLE1_GSK3B_REWARD_DISTANCE_WEIGHT:-0.05}"
RL_REWARD_SOURCE_SIMILARITY_WEIGHT="${SUCC_TABLE1_GSK3B_REWARD_SOURCE_SIMILARITY_WEIGHT:-2.0}"
RL_REWARD_SOURCE_SIMILARITY_THRESHOLD="${SUCC_TABLE1_GSK3B_REWARD_SOURCE_SIMILARITY_THRESHOLD:-0.5}"
RL_REWARD_SOURCE_COPY_PENALTY="${SUCC_TABLE1_GSK3B_REWARD_SOURCE_COPY_PENALTY:-1.0}"

BENCHMARK_NUM_SAMPLES="${SUCC_TABLE1_GSK3B_BENCHMARK_NUM_SAMPLES:-20}"
RUN_FEATURE_EXPORT="${SUCC_TABLE1_GSK3B_RUN_FEATURE_EXPORT:-auto}"
RUN_RL="${SUCC_TABLE1_GSK3B_RUN_RL:-1}"
RUN_BENCHMARK="${SUCC_TABLE1_GSK3B_RUN_BENCHMARK:-1}"

HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
HF_DEVICE_MAP="${SUCC_HF_DEVICE_MAP:-auto}"
HF_DTYPE="${SUCC_HF_DTYPE:-auto}"
HF_BATCH_SIZE="${SUCC_HF_BATCH_SIZE:-1}"
HF_MAX_LENGTH="${SUCC_HF_MAX_LENGTH:-2048}"
HF_RENDER_IMAGE_SIZE="${SUCC_HF_RENDER_IMAGE_SIZE:-256}"
POOLED_DIM="${SUCC_POOLED_DIM:-3584}"
NUM_QUERIES="${SUCC_NUM_QUERIES:-32}"
QUERY_DIM="${SUCC_QUERY_DIM:-256}"

export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-Unified-3MDiffusion:$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR" "$PACK_DIR" "$RL_MODEL_DIR" "$BENCHMARK_MODEL_DIR" "$BENCHMARK_OUTPUT_DIR"

echo "GSK3B n=20 no-rank group-RL pilot"
echo "  python=$PYTHON_BIN"
echo "  output_dir=$OUTPUT_DIR"
echo "  sft_checkpoint=$SFT_CHECKPOINT"
echo "  eval_limit=$EVAL_LIMIT"
echo "  rl_rollouts=$RL_ROLLOUTS_PER_PROMPT"
echo "  source_sim_weight=$RL_REWARD_SOURCE_SIMILARITY_WEIGHT"
echo "  source_sim_threshold=$RL_REWARD_SOURCE_SIMILARITY_THRESHOLD"
echo "  benchmark_num_samples=$BENCHMARK_NUM_SAMPLES"

if [[ ! -f "$SFT_CHECKPOINT" ]]; then
  echo "ERROR: missing SFT checkpoint: $SFT_CHECKPOINT" >&2
  exit 2
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/prepare_direct_smiles_table1_gsk3b_pilot_pack.py" \
  --train-condition-csv "$SOURCE_ROOT/table1_train_pack/table1_benchmark_condition_rows.csv" \
  --train-moledit-csv "$SOURCE_ROOT/table1_train_pack/table1_moledit_rows.csv" \
  --eval-reference-csv "$SOURCE_ROOT/benchmark_group_rl/direct_smiles_table1_selected_n20.csv" \
  --output-dir "$PACK_DIR" \
  --eval-limit "$EVAL_LIMIT"

TRAIN_ROWS_CSV="$PACK_DIR/table1_train_gsk3b_condition_rows.csv"
EVAL_ROWS_CSV="$PACK_DIR/table1_eval_gsk3b_condition_rows.csv"
EVAL_REFERENCE_CSV="$PACK_DIR/table1_eval_gsk3b_moledit_rows.csv"

export_eval_features() {
  local mode="$RUN_FEATURE_EXPORT"
  local should_export=0
  if [[ "$mode" == "1" || "$mode" == "true" || "$mode" == "yes" ]]; then
    should_export=1
  elif [[ "$mode" == "auto" && ! -f "$EVAL_FEATURES_DIR/query_tokens.npy" ]]; then
    should_export=1
  fi
  if [[ "$should_export" == "1" ]]; then
    export SUCC_ENCODER=hf_vlm
    export SUCC_VARIANTS=full
    export SUCC_BASELINE_CSV="$EVAL_ROWS_CSV"
    export SUCC_OUTPUT_DIR="$EVAL_FEATURES_DIR"
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

export_eval_features

FEATURE_ARGS=(--condition-features-dir "$TRAIN_FEATURES_DIR")
if [[ -f "$EVAL_FEATURES_DIR/query_tokens.npy" ]]; then
  FEATURE_ARGS+=(--eval-condition-features-dir "$EVAL_FEATURES_DIR")
fi

if [[ "$RUN_RL" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator_rl.py" \
    --train-csv "$TRAIN_ROWS_CSV" \
    --eval-csv "$EVAL_ROWS_CSV" \
    --output-dir "$RL_MODEL_DIR" \
    --resume-checkpoint "$SFT_CHECKPOINT" \
    "${FEATURE_ARGS[@]}" \
    --condition-mixing-mode "$CONDITION_MIXING_MODE" \
    --epochs "$RL_EPOCHS" \
    --batch-size "$RL_BATCH_SIZE" \
    --eval-batch-size "$RL_EVAL_BATCH_SIZE" \
    --lr "$RL_LR" \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
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
    --advantage-clip 3.0 \
    --sequence-logprob-reduction mean \
    --reference-kl-weight "$RL_REFERENCE_KL_WEIGHT" \
    --reward-mode table1_edit \
    --reward-valid-weight "$RL_REWARD_VALID_WEIGHT" \
    --reward-strict-weight "$RL_REWARD_STRICT_WEIGHT" \
    --reward-distance-weight "$RL_REWARD_DISTANCE_WEIGHT" \
    --reward-distance-clip 10.0 \
    --reward-source-similarity-weight "$RL_REWARD_SOURCE_SIMILARITY_WEIGHT" \
    --reward-source-similarity-threshold "$RL_REWARD_SOURCE_SIMILARITY_THRESHOLD" \
    --reward-source-copy-penalty "$RL_REWARD_SOURCE_COPY_PENALTY" \
    --seed "$SEED" \
    --device "$DEVICE"
fi

if [[ ! -f "$RL_CHECKPOINT" ]]; then
  echo "ERROR: missing RL checkpoint: $RL_CHECKPOINT" >&2
  exit 2
fi

if [[ "$RUN_BENCHMARK" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py" \
    --eval-only \
    --eval-csv "$EVAL_ROWS_CSV" \
    --resume-checkpoint "$RL_CHECKPOINT" \
    "${FEATURE_ARGS[@]}" \
    --condition-mixing-mode "$CONDITION_MIXING_MODE" \
    --output-dir "$BENCHMARK_MODEL_DIR" \
    --prediction-csv "$RAW_SELECTED_CSV" \
    --candidate-output-csv "$CANDIDATE_PREDICTION_CSV" \
    --eval-batch-size "$RL_EVAL_BATCH_SIZE" \
    --max-new-tokens "$RL_MAX_NEW_TOKENS" \
    --temperature "$RL_TEMPERATURE" \
    --top-k "$RL_TOP_K" \
    --top-p "$RL_TOP_P" \
    --num-samples "$BENCHMARK_NUM_SAMPLES" \
    --parallel-samples "$RL_PARALLEL_SAMPLES" \
    --max-parallel-sequences "$RL_MAX_PARALLEL_SEQUENCES" \
    --repetition-penalty "$RL_REPETITION_PENALTY" \
    --no-repeat-ngram-size "$RL_NO_REPEAT_NGRAM_SIZE" \
    --min-new-tokens "$RL_MIN_NEW_TOKENS" \
    --disable-property-rerank \
    --seed "$SEED" \
    --device "$DEVICE"

  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$EVAL_REFERENCE_CSV" \
    --candidates "$CANDIDATE_PREDICTION_CSV" \
    --output-dir "$BENCHMARK_OUTPUT_DIR/moledit_table_metrics_any20" \
    --candidate-limit "$BENCHMARK_NUM_SAMPLES" \
    --model-name "DirectSMILES-GSK3B-n20-any" \
    --thresholds "0.65,0.15" \
    --task-filter table1 \
    --missing-oracle-policy fail
fi

echo
echo "GSK3B n=20 pilot ready:"
echo "  train_rows=$TRAIN_ROWS_CSV"
echo "  eval_rows=$EVAL_ROWS_CSV"
echo "  rl_checkpoint=$RL_CHECKPOINT"
echo "  candidate_predictions=$CANDIDATE_PREDICTION_CSV"
echo "  any20_markdown=$BENCHMARK_OUTPUT_DIR/moledit_table_metrics_any20/moledit_table_summary.md"
