#!/usr/bin/env bash
# Run source-conditioned direct-SMILES group RL on MolEdit Table1 tasks.
#
# This mirrors the de novo group-RL recipe:
#   SFT warm-start -> group-relative RL -> small-n candidate evaluation.

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
DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
OUTPUT_DIR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_moledit_table1_group_rl_v1}"
MOLEDIT_TRAIN_SPLIT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_TRAIN_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
MOLEDIT_EVAL_SPLIT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
TRAIN_PACK_DIR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_TRAIN_PACK_DIR:-$OUTPUT_DIR/table1_train_pack}"
EVAL_PACK_DIR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_EVAL_PACK_DIR:-$OUTPUT_DIR/table1_eval_pack}"
TRAIN_ROWS_CSV="${SUCC_DIRECT_MOLEDIT_GROUP_RL_TRAIN_ROWS_CSV:-$TRAIN_PACK_DIR/table1_benchmark_condition_rows.csv}"
EVAL_ROWS_CSV="${SUCC_DIRECT_MOLEDIT_GROUP_RL_EVAL_ROWS_CSV:-$EVAL_PACK_DIR/table1_benchmark_condition_rows.csv}"
EVAL_REFERENCE_CSV="${SUCC_DIRECT_MOLEDIT_GROUP_RL_EVAL_REFERENCE_CSV:-$EVAL_PACK_DIR/table1_moledit_rows.csv}"
TRAIN_FEATURES_DIR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_TRAIN_FEATURES_DIR:-$OUTPUT_DIR/train_condition_features_hf_vlm}"
EVAL_FEATURES_DIR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_EVAL_FEATURES_DIR:-$OUTPUT_DIR/eval_condition_features_hf_vlm}"
SFT_MODEL_DIR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SFT_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_source_edit_sft}"
RL_MODEL_DIR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_group_rl}"
BENCHMARK_MODEL_DIR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_group_rl_eval}"
BENCHMARK_OUTPUT_DIR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_group_rl}"
CANDIDATE_PREDICTION_CSV="${SUCC_DIRECT_MOLEDIT_GROUP_RL_CANDIDATE_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_candidate_predictions.csv}"
RAW_SELECTED_CSV="${SUCC_DIRECT_MOLEDIT_GROUP_RL_RAW_SELECTED_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_selected_raw.csv}"

RUN_PACK_EXPORT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_RUN_PACK_EXPORT:-auto}"
RUN_FEATURE_EXPORT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_RUN_FEATURE_EXPORT:-auto}"
RUN_SFT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_RUN_SFT:-1}"
RUN_RL="${SUCC_DIRECT_MOLEDIT_GROUP_RL_RUN_RL:-1}"
RUN_BENCHMARK="${SUCC_DIRECT_MOLEDIT_GROUP_RL_RUN_BENCHMARK:-1}"
TRAIN_PER_TASK="${SUCC_DIRECT_MOLEDIT_GROUP_RL_TRAIN_PER_TASK:-500}"
EVAL_PER_TASK="${SUCC_DIRECT_MOLEDIT_GROUP_RL_EVAL_PER_TASK:-100}"
SYNTHESIZE_MISSING_TASKS="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SYNTHESIZE_MISSING_TASKS:-1}"
SYNTHETIC_MIN_SOURCE_TANIMOTO="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SYNTHETIC_MIN_SOURCE_TANIMOTO:-0.4}"
SYNTHETIC_CANDIDATE_LIMIT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SYNTHETIC_CANDIDATE_LIMIT:-8000}"

BASE_CHECKPOINT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BASE_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt}"
SFT_CHECKPOINT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SFT_CHECKPOINT:-$SFT_MODEL_DIR/direct_smiles_generator.pt}"
RL_CHECKPOINT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_RL_CHECKPOINT:-$RL_MODEL_DIR/direct_smiles_generator_rl.pt}"
CONDITION_MIXING_MODE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_CONDITION_MIXING_MODE:-append_source_property_program}"
DEVICE="${SUCC_DEVICE:-auto}"
SEED="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SEED:-23}"

SFT_EPOCHS="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SFT_EPOCHS:-1}"
SFT_BATCH_SIZE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SFT_BATCH_SIZE:-32}"
SFT_EVAL_BATCH_SIZE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SFT_EVAL_BATCH_SIZE:-32}"
SFT_LR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SFT_LR:-1e-5}"
SFT_WEIGHT_DECAY="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SFT_WEIGHT_DECAY:-1e-4}"
SFT_GRAD_CLIP="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SFT_GRAD_CLIP:-1.0}"

RL_EPOCHS="${SUCC_DIRECT_MOLEDIT_GROUP_RL_EPOCHS:-1}"
RL_BATCH_SIZE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BATCH_SIZE:-8}"
RL_EVAL_BATCH_SIZE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_EVAL_BATCH_SIZE:-16}"
RL_LR="${SUCC_DIRECT_MOLEDIT_GROUP_RL_LR:-5e-7}"
RL_WEIGHT_DECAY="${SUCC_DIRECT_MOLEDIT_GROUP_RL_WEIGHT_DECAY:-1e-4}"
RL_GRAD_CLIP="${SUCC_DIRECT_MOLEDIT_GROUP_RL_GRAD_CLIP:-1.0}"
RL_ROLLOUTS_PER_PROMPT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_ROLLOUTS_PER_PROMPT:-16}"
RL_PARALLEL_SAMPLES="${SUCC_DIRECT_MOLEDIT_GROUP_RL_PARALLEL_SAMPLES:-4}"
RL_MAX_PARALLEL_SEQUENCES="${SUCC_DIRECT_MOLEDIT_GROUP_RL_MAX_PARALLEL_SEQUENCES:-512}"
RL_MAX_NEW_TOKENS="${SUCC_DIRECT_MOLEDIT_GROUP_RL_MAX_NEW_TOKENS:-100}"
RL_TEMPERATURE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_TEMPERATURE:-0.85}"
RL_TOP_K="${SUCC_DIRECT_MOLEDIT_GROUP_RL_TOP_K:-40}"
RL_TOP_P="${SUCC_DIRECT_MOLEDIT_GROUP_RL_TOP_P:-0.95}"
RL_REPETITION_PENALTY="${SUCC_DIRECT_MOLEDIT_GROUP_RL_REPETITION_PENALTY:-1.15}"
RL_NO_REPEAT_NGRAM_SIZE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_NO_REPEAT_NGRAM_SIZE:-6}"
RL_MIN_NEW_TOKENS="${SUCC_DIRECT_MOLEDIT_GROUP_RL_MIN_NEW_TOKENS:-6}"
RL_SFT_WEIGHT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SFT_WEIGHT:-1.0}"
RL_ADVANTAGE_MODE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_ADVANTAGE_MODE:-group_zscore}"
RL_ADVANTAGE_CLIP="${SUCC_DIRECT_MOLEDIT_GROUP_RL_ADVANTAGE_CLIP:-3.0}"
RL_SEQUENCE_LOGPROB_REDUCTION="${SUCC_DIRECT_MOLEDIT_GROUP_RL_SEQUENCE_LOGPROB_REDUCTION:-mean}"
RL_REFERENCE_KL_WEIGHT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_REFERENCE_KL_WEIGHT:-0.05}"
RL_REWARD_VALID_WEIGHT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_REWARD_VALID_WEIGHT:-0.25}"
RL_REWARD_STRICT_WEIGHT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_REWARD_STRICT_WEIGHT:-2.0}"
RL_REWARD_DISTANCE_WEIGHT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_REWARD_DISTANCE_WEIGHT:-0.05}"
RL_REWARD_DISTANCE_CLIP="${SUCC_DIRECT_MOLEDIT_GROUP_RL_REWARD_DISTANCE_CLIP:-10.0}"
RL_REWARD_SOURCE_SIMILARITY_WEIGHT="${SUCC_DIRECT_MOLEDIT_GROUP_RL_REWARD_SOURCE_SIMILARITY_WEIGHT:-0.75}"
RL_REWARD_SOURCE_SIMILARITY_THRESHOLD="${SUCC_DIRECT_MOLEDIT_GROUP_RL_REWARD_SOURCE_SIMILARITY_THRESHOLD:-0.4}"
RL_REWARD_SOURCE_COPY_PENALTY="${SUCC_DIRECT_MOLEDIT_GROUP_RL_REWARD_SOURCE_COPY_PENALTY:-0.5}"

BENCHMARK_MAX_NEW_TOKENS="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_MAX_NEW_TOKENS:-100}"
BENCHMARK_TEMPERATURE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_TEMPERATURE:-0.85}"
BENCHMARK_TOP_K="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_TOP_K:-40}"
BENCHMARK_TOP_P="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_TOP_P:-0.95}"
BENCHMARK_NUM_SAMPLES="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_NUM_SAMPLES:-256}"
BENCHMARK_BUDGETS_RAW="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_BUDGETS:-20 256}"
BENCHMARK_PARALLEL_SAMPLES="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_PARALLEL_SAMPLES:-4}"
BENCHMARK_MAX_PARALLEL_SEQUENCES="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_MAX_PARALLEL_SEQUENCES:-512}"
BENCHMARK_REPETITION_PENALTY="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_REPETITION_PENALTY:-1.15}"
BENCHMARK_NO_REPEAT_NGRAM_SIZE="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_NO_REPEAT_NGRAM_SIZE:-6}"
BENCHMARK_MIN_NEW_TOKENS="${SUCC_DIRECT_MOLEDIT_GROUP_RL_BENCHMARK_MIN_NEW_TOKENS:-6}"

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

mkdir -p "$OUTPUT_DIR" "$SFT_MODEL_DIR" "$RL_MODEL_DIR" "$BENCHMARK_MODEL_DIR" "$BENCHMARK_OUTPUT_DIR"

echo "Direct-SMILES MolEdit Table1 source-conditioned group RL"
echo "  python=$PYTHON_BIN"
echo "  output_dir=$OUTPUT_DIR"
echo "  train_split=$MOLEDIT_TRAIN_SPLIT"
echo "  eval_split=$MOLEDIT_EVAL_SPLIT"
echo "  base_checkpoint=$BASE_CHECKPOINT"
echo "  sft_checkpoint=$SFT_CHECKPOINT"
echo "  rl_checkpoint=$RL_CHECKPOINT"
echo "  condition_mixing_mode=$CONDITION_MIXING_MODE"
echo "  run_sft=$RUN_SFT"
echo "  run_rl=$RUN_RL"
echo "  run_benchmark=$RUN_BENCHMARK"
echo "  rl_rollouts_per_prompt=$RL_ROLLOUTS_PER_PROMPT"
echo "  rl_advantage_mode=$RL_ADVANTAGE_MODE"
echo "  rl_reference_kl_weight=$RL_REFERENCE_KL_WEIGHT"
echo "  benchmark_num_samples=$BENCHMARK_NUM_SAMPLES"
echo "  benchmark_budgets=$BENCHMARK_BUDGETS_RAW"

export_pack() {
  local output_dir="$1"
  local per_task="$2"
  local eval_first="$3"
  local mode="$RUN_PACK_EXPORT"
  local should_export=0
  if [[ "$mode" == "1" || "$mode" == "true" || "$mode" == "yes" ]]; then
    should_export=1
  elif [[ "$mode" == "auto" && ! -f "$output_dir/table1_benchmark_condition_rows.csv" ]]; then
    should_export=1
  elif [[ "$mode" != "0" && "$mode" != "false" && "$mode" != "no" && "$mode" != "auto" ]]; then
    echo "ERROR: unsupported SUCC_DIRECT_MOLEDIT_GROUP_RL_RUN_PACK_EXPORT=$mode" >&2
    exit 2
  fi
  if [[ "$should_export" == "1" ]]; then
    PACK_ARGS=(
      "$PYTHON_BIN"
      "$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts/export_moledit_table1_benchmark_pack.py"
      --moledit-train-split "$MOLEDIT_TRAIN_SPLIT"
      --moledit-eval-split "$MOLEDIT_EVAL_SPLIT"
      --output-dir "$output_dir"
      --per-task "$per_task"
      --synthetic-min-source-tanimoto "$SYNTHETIC_MIN_SOURCE_TANIMOTO"
      --synthetic-candidate-limit "$SYNTHETIC_CANDIDATE_LIMIT"
    )
    if [[ "$eval_first" == "1" ]]; then
      PACK_ARGS+=(--eval-first)
    else
      PACK_ARGS+=(--no-eval-first)
    fi
    if [[ "$SYNTHESIZE_MISSING_TASKS" == "1" ]]; then
      PACK_ARGS+=(--synthesize-missing-tasks)
    fi
    "${PACK_ARGS[@]}"
  fi
}

export_features() {
  local rows_csv="$1"
  local features_dir="$2"
  local mode="$RUN_FEATURE_EXPORT"
  local should_export=0
  if [[ "$mode" == "1" || "$mode" == "true" || "$mode" == "yes" ]]; then
    should_export=1
  elif [[ "$mode" == "auto" && ! -f "$features_dir/query_tokens.npy" ]]; then
    should_export=1
  elif [[ "$mode" != "0" && "$mode" != "false" && "$mode" != "no" && "$mode" != "auto" ]]; then
    echo "ERROR: unsupported SUCC_DIRECT_MOLEDIT_GROUP_RL_RUN_FEATURE_EXPORT=$mode" >&2
    exit 2
  fi
  if [[ "$should_export" == "1" ]]; then
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

export_pack "$TRAIN_PACK_DIR" "$TRAIN_PER_TASK" 0
export_pack "$EVAL_PACK_DIR" "$EVAL_PER_TASK" 1
for required in "$TRAIN_ROWS_CSV" "$EVAL_ROWS_CSV" "$EVAL_REFERENCE_CSV"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing required file: $required" >&2
    exit 2
  fi
done

export_features "$TRAIN_ROWS_CSV" "$TRAIN_FEATURES_DIR"
export_features "$EVAL_ROWS_CSV" "$EVAL_FEATURES_DIR"

FEATURE_ARGS=()
if [[ -f "$TRAIN_FEATURES_DIR/query_tokens.npy" ]]; then
  FEATURE_ARGS+=(--condition-features-dir "$TRAIN_FEATURES_DIR")
fi
if [[ -f "$EVAL_FEATURES_DIR/query_tokens.npy" ]]; then
  FEATURE_ARGS+=(--eval-condition-features-dir "$EVAL_FEATURES_DIR")
fi

if [[ "$RUN_SFT" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py" \
    --train-csv "$TRAIN_ROWS_CSV" \
    --eval-csv "$EVAL_ROWS_CSV" \
    --resume-checkpoint "$BASE_CHECKPOINT" \
    --reset-training-state \
    "${FEATURE_ARGS[@]}" \
    --condition-mixing-mode "$CONDITION_MIXING_MODE" \
    --output-dir "$SFT_MODEL_DIR" \
    --epochs "$SFT_EPOCHS" \
    --batch-size "$SFT_BATCH_SIZE" \
    --eval-batch-size "$SFT_EVAL_BATCH_SIZE" \
    --lr "$SFT_LR" \
    --weight-decay "$SFT_WEIGHT_DECAY" \
    --grad-clip "$SFT_GRAD_CLIP" \
    --max-new-tokens "$BENCHMARK_MAX_NEW_TOKENS" \
    --temperature 0.70 \
    --top-k 24 \
    --top-p 0.90 \
    --num-samples 1 \
    --seed "$SEED" \
    --device "$DEVICE"
else
  echo "Skipping SFT warm-start (RUN_SFT=$RUN_SFT)"
fi

if [[ ! -f "$SFT_CHECKPOINT" ]]; then
  echo "ERROR: missing SFT checkpoint: $SFT_CHECKPOINT" >&2
  exit 2
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
    --reward-mode table1_edit \
    --reward-valid-weight "$RL_REWARD_VALID_WEIGHT" \
    --reward-strict-weight "$RL_REWARD_STRICT_WEIGHT" \
    --reward-distance-weight "$RL_REWARD_DISTANCE_WEIGHT" \
    --reward-distance-clip "$RL_REWARD_DISTANCE_CLIP" \
    --reward-source-similarity-weight "$RL_REWARD_SOURCE_SIMILARITY_WEIGHT" \
    --reward-source-similarity-threshold "$RL_REWARD_SOURCE_SIMILARITY_THRESHOLD" \
    --reward-source-copy-penalty "$RL_REWARD_SOURCE_COPY_PENALTY" \
    --seed "$SEED" \
    --device "$DEVICE"
else
  echo "Skipping group RL training (RUN_RL=$RUN_RL)"
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
    --disable-property-rerank \
    --seed "$SEED" \
    --device "$DEVICE"

  BUDGETS=()
  for budget in ${BENCHMARK_BUDGETS_RAW//,/ }; do
    if [[ -n "$budget" ]]; then
      BUDGETS+=("$budget")
    fi
  done
  for budget in "${BUDGETS[@]}"; do
    selected_csv="$BENCHMARK_OUTPUT_DIR/direct_smiles_table1_selected_n${budget}.csv"
    table_dir="$BENCHMARK_OUTPUT_DIR/moledit_table_metrics_n${budget}"
    method_name="direct_smiles_moledit_table1_group_rl_n${budget}"
    "$PYTHON_BIN" "$PROJECT_DIR/scripts/select_moledit_table1_direct_smiles_candidates.py" \
      --reference "$EVAL_REFERENCE_CSV" \
      --candidate-predictions "$CANDIDATE_PREDICTION_CSV" \
      --output-csv "$selected_csv" \
      --candidate-limit "$budget" \
      --method-name "$method_name" \
      --source-similarity-threshold "$RL_REWARD_SOURCE_SIMILARITY_THRESHOLD"
    "$PYTHON_BIN" "$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts/evaluate_moledit_table_metrics.py" \
      --reference "$EVAL_REFERENCE_CSV" \
      --predictions "$selected_csv" \
      --method "$method_name" \
      --output-dir "$table_dir" \
      --model-name "DirectSMILES-EditGroupRL" \
      --thresholds "0.65,0.15" \
      --task-filter table1 \
      --include-empty-table1 \
      --require-table1-coverage \
      --missing-oracle-policy fail
  done
fi

echo
echo "Direct-SMILES MolEdit Table1 group RL ready:"
echo "  train_rows=$TRAIN_ROWS_CSV"
echo "  eval_rows=$EVAL_ROWS_CSV"
echo "  sft_checkpoint=$SFT_CHECKPOINT"
echo "  rl_checkpoint=$RL_CHECKPOINT"
echo "  candidate_predictions=$CANDIDATE_PREDICTION_CSV"
echo "  benchmark_output_dir=$BENCHMARK_OUTPUT_DIR"
