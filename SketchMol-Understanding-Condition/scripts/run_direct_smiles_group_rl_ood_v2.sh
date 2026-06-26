#!/usr/bin/env bash
# Run a group-relative RL fine-tune for the v2 direct-SMILES OOD model.

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
BASE_OUTPUT_DIR="${SUCC_DIRECT_OOD_GROUP_RL_BASE_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_ood_v2_mixed_condition}"
OUTPUT_DIR="${SUCC_DIRECT_OOD_GROUP_RL_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_ood_v2_group_rl_v1}"
TRAIN_ROWS_CSV="${SUCC_DIRECT_OOD_GROUP_RL_TRAIN_ROWS_CSV:-$BASE_OUTPUT_DIR/denovo_ood_train_rows.csv}"
EVAL_ROWS_CSV="${SUCC_DIRECT_OOD_GROUP_RL_EVAL_ROWS_CSV:-$BASE_OUTPUT_DIR/denovo_ood_eval_rows.csv}"
TRAIN_FEATURES_DIR="${SUCC_DIRECT_OOD_GROUP_RL_TRAIN_FEATURES_DIR:-$BASE_OUTPUT_DIR/train_condition_features_hf_vlm}"
EVAL_FEATURES_DIR="${SUCC_DIRECT_OOD_GROUP_RL_EVAL_FEATURES_DIR:-$BASE_OUTPUT_DIR/eval_condition_features_hf_vlm}"
RESUME_CHECKPOINT="${SUCC_DIRECT_OOD_GROUP_RL_RESUME_CHECKPOINT:-$BASE_OUTPUT_DIR/direct_smiles_model/direct_smiles_generator.pt}"
MODEL_DIR="${SUCC_DIRECT_OOD_GROUP_RL_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_group_rl}"
RUN_TRAIN="${SUCC_DIRECT_OOD_GROUP_RL_RUN_TRAIN:-1}"
RUN_BENCHMARK_AFTER_TRAIN="${SUCC_DIRECT_OOD_GROUP_RL_RUN_BENCHMARK_AFTER_TRAIN:-1}"
BENCHMARK_OUTPUT_DIR="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_OUTPUT_DIR:-$OUTPUT_DIR/benchmark_direct_smiles_group_rl}"
BENCHMARK_MODEL_DIR="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_MODEL_DIR:-$OUTPUT_DIR/direct_smiles_model_group_rl_eval}"
BENCHMARK_PREDICTION_CSV="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_PREDICTION_CSV:-$BENCHMARK_OUTPUT_DIR/direct_smiles_predictions.csv}"
CONDITION_MIXING_MODE="${SUCC_DIRECT_OOD_GROUP_RL_CONDITION_MIXING_MODE:-append_property_program}"

RL_EPOCHS="${SUCC_DIRECT_OOD_GROUP_RL_EPOCHS:-1}"
RL_BATCH_SIZE="${SUCC_DIRECT_OOD_GROUP_RL_BATCH_SIZE:-8}"
RL_EVAL_BATCH_SIZE="${SUCC_DIRECT_OOD_GROUP_RL_EVAL_BATCH_SIZE:-32}"
RL_LR="${SUCC_DIRECT_OOD_GROUP_RL_LR:-1e-6}"
RL_WEIGHT_DECAY="${SUCC_DIRECT_OOD_GROUP_RL_WEIGHT_DECAY:-1e-4}"
RL_GRAD_CLIP="${SUCC_DIRECT_OOD_GROUP_RL_GRAD_CLIP:-1.0}"
RL_ROLLOUTS_PER_PROMPT="${SUCC_DIRECT_OOD_GROUP_RL_ROLLOUTS_PER_PROMPT:-16}"
RL_PARALLEL_SAMPLES="${SUCC_DIRECT_OOD_GROUP_RL_PARALLEL_SAMPLES:-4}"
RL_MAX_PARALLEL_SEQUENCES="${SUCC_DIRECT_OOD_GROUP_RL_MAX_PARALLEL_SEQUENCES:-512}"
RL_MAX_NEW_TOKENS="${SUCC_DIRECT_OOD_GROUP_RL_MAX_NEW_TOKENS:-96}"
RL_TEMPERATURE="${SUCC_DIRECT_OOD_GROUP_RL_TEMPERATURE:-0.85}"
RL_TOP_K="${SUCC_DIRECT_OOD_GROUP_RL_TOP_K:-40}"
RL_TOP_P="${SUCC_DIRECT_OOD_GROUP_RL_TOP_P:-0.95}"
RL_REPETITION_PENALTY="${SUCC_DIRECT_OOD_GROUP_RL_REPETITION_PENALTY:-1.15}"
RL_NO_REPEAT_NGRAM_SIZE="${SUCC_DIRECT_OOD_GROUP_RL_NO_REPEAT_NGRAM_SIZE:-6}"
RL_MIN_NEW_TOKENS="${SUCC_DIRECT_OOD_GROUP_RL_MIN_NEW_TOKENS:-6}"
RL_SFT_WEIGHT="${SUCC_DIRECT_OOD_GROUP_RL_SFT_WEIGHT:-1.0}"
RL_ADVANTAGE_MODE="${SUCC_DIRECT_OOD_GROUP_RL_ADVANTAGE_MODE:-group_zscore}"
RL_ADVANTAGE_CLIP="${SUCC_DIRECT_OOD_GROUP_RL_ADVANTAGE_CLIP:-3.0}"
RL_SEQUENCE_LOGPROB_REDUCTION="${SUCC_DIRECT_OOD_GROUP_RL_SEQUENCE_LOGPROB_REDUCTION:-mean}"
RL_REFERENCE_KL_WEIGHT="${SUCC_DIRECT_OOD_GROUP_RL_REFERENCE_KL_WEIGHT:-0.05}"
RL_REWARD_VALID_WEIGHT="${SUCC_DIRECT_OOD_GROUP_RL_REWARD_VALID_WEIGHT:-0.25}"
RL_REWARD_STRICT_WEIGHT="${SUCC_DIRECT_OOD_GROUP_RL_REWARD_STRICT_WEIGHT:-2.0}"
RL_REWARD_DISTANCE_WEIGHT="${SUCC_DIRECT_OOD_GROUP_RL_REWARD_DISTANCE_WEIGHT:-0.05}"
RL_REWARD_DISTANCE_CLIP="${SUCC_DIRECT_OOD_GROUP_RL_REWARD_DISTANCE_CLIP:-10.0}"
DEVICE="${SUCC_DEVICE:-auto}"
SEED="${SUCC_DIRECT_OOD_GROUP_RL_SEED:-7}"

BENCHMARK_MAX_NEW_TOKENS="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_MAX_NEW_TOKENS:-96}"
BENCHMARK_TEMPERATURE="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_TEMPERATURE:-0.85}"
BENCHMARK_TOP_K="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_TOP_K:-40}"
BENCHMARK_TOP_P="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_TOP_P:-0.95}"
BENCHMARK_NUM_SAMPLES="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_NUM_SAMPLES:-128}"
BENCHMARK_PARALLEL_SAMPLES="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_PARALLEL_SAMPLES:-8}"
BENCHMARK_MAX_PARALLEL_SEQUENCES="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_MAX_PARALLEL_SEQUENCES:-1024}"
BENCHMARK_REPETITION_PENALTY="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_REPETITION_PENALTY:-1.15}"
BENCHMARK_NO_REPEAT_NGRAM_SIZE="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_NO_REPEAT_NGRAM_SIZE:-6}"
BENCHMARK_MIN_NEW_TOKENS="${SUCC_DIRECT_OOD_GROUP_RL_BENCHMARK_MIN_NEW_TOKENS:-6}"

mkdir -p "$OUTPUT_DIR" "$MODEL_DIR" "$BENCHMARK_OUTPUT_DIR" "$BENCHMARK_MODEL_DIR"

echo "Direct-SMILES group-relative RL v2 (OOD)"
echo "  python=$PYTHON_BIN"
echo "  base_output_dir=$BASE_OUTPUT_DIR"
echo "  output_dir=$OUTPUT_DIR"
echo "  resume_checkpoint=$RESUME_CHECKPOINT"
echo "  condition_mixing_mode=$CONDITION_MIXING_MODE"
echo "  rl_epochs=$RL_EPOCHS"
echo "  rl_rollouts_per_prompt=$RL_ROLLOUTS_PER_PROMPT"
echo "  rl_lr=$RL_LR"
echo "  rl_sft_weight=$RL_SFT_WEIGHT"
echo "  rl_advantage_mode=$RL_ADVANTAGE_MODE"
echo "  rl_advantage_clip=$RL_ADVANTAGE_CLIP"
echo "  rl_sequence_logprob_reduction=$RL_SEQUENCE_LOGPROB_REDUCTION"
echo "  rl_reference_kl_weight=$RL_REFERENCE_KL_WEIGHT"
echo "  reward_valid_weight=$RL_REWARD_VALID_WEIGHT"
echo "  reward_strict_weight=$RL_REWARD_STRICT_WEIGHT"
echo "  reward_distance_weight=$RL_REWARD_DISTANCE_WEIGHT"
echo "  run_train=$RUN_TRAIN"
echo "  run_benchmark_after_train=$RUN_BENCHMARK_AFTER_TRAIN"
echo "  benchmark_num_samples=$BENCHMARK_NUM_SAMPLES"

if [[ "$RUN_TRAIN" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator_rl.py" \
    --train-csv "$TRAIN_ROWS_CSV" \
    --eval-csv "$EVAL_ROWS_CSV" \
    --output-dir "$MODEL_DIR" \
    --resume-checkpoint "$RESUME_CHECKPOINT" \
    --condition-features-dir "$TRAIN_FEATURES_DIR" \
    --eval-condition-features-dir "$EVAL_FEATURES_DIR" \
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
    --seed "$SEED" \
    --device "$DEVICE"
else
  echo "Skipping RL training (SUCC_DIRECT_OOD_GROUP_RL_RUN_TRAIN=0)"
fi

RL_CHECKPOINT="${SUCC_DIRECT_OOD_GROUP_RL_RL_CHECKPOINT:-$MODEL_DIR/direct_smiles_generator_rl.pt}"
if [[ ! -f "$RL_CHECKPOINT" ]]; then
  echo "ERROR: RL checkpoint not found: $RL_CHECKPOINT" >&2
  exit 1
fi
echo "  rl_checkpoint=$RL_CHECKPOINT"

if [[ "$RUN_BENCHMARK_AFTER_TRAIN" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py" \
    --eval-only \
    --eval-csv "$EVAL_ROWS_CSV" \
    --resume-checkpoint "$RL_CHECKPOINT" \
    --condition-features-dir "$TRAIN_FEATURES_DIR" \
    --eval-condition-features-dir "$EVAL_FEATURES_DIR" \
    --condition-mixing-mode "$CONDITION_MIXING_MODE" \
    --output-dir "$BENCHMARK_MODEL_DIR" \
    --prediction-csv "$BENCHMARK_PREDICTION_CSV" \
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
    --seed "$SEED" \
    --device "$DEVICE"

  "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_univideo_image_benchmark.py" \
    --image-csv "$BENCHMARK_PREDICTION_CSV" \
    --output-dir "$BENCHMARK_OUTPUT_DIR" \
    --method direct_smiles_mllm \
    --smiles-column generated_smiles \
    --report-title "SUCC Direct SMILES Group-RL v2 OOD Benchmark" \
    --benchmark-family "direct_smiles_denovo_ood_property_design" \
    --benchmark-task "direct_smiles_denovo_ood_property_design" \
    --accept-direct-smiles \
    --hide-source-similarity-section

  echo
  echo "Direct-SMILES group-relative RL OOD benchmark ready:"
  echo "  report=$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
  echo "  summary=$BENCHMARK_OUTPUT_DIR/benchmark_summary.csv"
  echo "  prediction_csv=$BENCHMARK_PREDICTION_CSV"
  sed -n '1,100p' "$BENCHMARK_OUTPUT_DIR/benchmark_report.md"
fi
