#!/usr/bin/env bash
# Generate one frozen n=256 P1 candidate pool without retraining.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
VARIANT="${P1_VARIANT:-${1:-}}"
SEED="${SUCC_P1_SEED:-7}"
NUM_SAMPLES="${SUCC_P1_NUM_SAMPLES:-256}"
OUTPUT_ROOT="${SUCC_P1_OUTPUT_ROOT:-$PROJECT_DIR/outputs/p1_property_program_group_rl_seed7}"

if [[ "$NUM_SAMPLES" != "256" ]]; then
  echo "ERROR: P1 preregistration fixes num_samples=256; got $NUM_SAMPLES" >&2
  exit 2
fi

case "$VARIANT" in
  two_p_to_seven_p_sft)
    BASE_DIR="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
    CHECKPOINT="$BASE_DIR/direct_smiles_model/direct_smiles_generator.pt"
    EVAL_CSV="$BASE_DIR/denovo_2p7p_eval_rows.csv"
    TRAIN_FEATURES="$BASE_DIR/train_condition_features_hf_vlm"
    EVAL_FEATURES="$BASE_DIR/eval_condition_features_hf_vlm"
    TEMPERATURE="0.85"
    TOP_K="40"
    TOP_P="0.95"
    REPETITION_PENALTY="1.15"
    ;;
  two_p_to_seven_p_group_rl)
    BASE_DIR="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"
    CHECKPOINT="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt"
    EVAL_CSV="$BASE_DIR/denovo_2p7p_eval_rows.csv"
    TRAIN_FEATURES="$BASE_DIR/train_condition_features_hf_vlm"
    EVAL_FEATURES="$BASE_DIR/eval_condition_features_hf_vlm"
    TEMPERATURE="0.85"
    TOP_K="40"
    TOP_P="0.95"
    REPETITION_PENALTY="1.15"
    ;;
  ood_sft)
    BASE_DIR="$PROJECT_DIR/outputs/direct_smiles_denovo_ood_v2_mixed_condition"
    CHECKPOINT="$BASE_DIR/direct_smiles_model/direct_smiles_generator.pt"
    EVAL_CSV="$BASE_DIR/denovo_ood_eval_rows.csv"
    TRAIN_FEATURES="$BASE_DIR/train_condition_features_hf_vlm"
    EVAL_FEATURES="$BASE_DIR/eval_condition_features_hf_vlm"
    TEMPERATURE="0.70"
    TOP_K="24"
    TOP_P="0.90"
    REPETITION_PENALTY="1.20"
    ;;
  ood_group_rl)
    BASE_DIR="$PROJECT_DIR/outputs/direct_smiles_denovo_ood_v2_mixed_condition"
    CHECKPOINT="$PROJECT_DIR/outputs/direct_smiles_denovo_ood_v2_group_rl_v1/direct_smiles_model_group_rl/direct_smiles_generator_rl.pt"
    EVAL_CSV="$BASE_DIR/denovo_ood_eval_rows.csv"
    TRAIN_FEATURES="$BASE_DIR/train_condition_features_hf_vlm"
    EVAL_FEATURES="$BASE_DIR/eval_condition_features_hf_vlm"
    TEMPERATURE="0.70"
    TOP_K="24"
    TOP_P="0.90"
    REPETITION_PENALTY="1.20"
    ;;
  *)
    echo "Usage: P1_VARIANT={two_p_to_seven_p_sft|two_p_to_seven_p_group_rl|ood_sft|ood_group_rl} $0" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="$OUTPUT_ROOT/$VARIANT"
CANDIDATE_CSV="$OUTPUT_DIR/raw_candidates_n256.csv"
DIAGNOSTIC_SELECTED_CSV="$OUTPUT_DIR/diagnostic_property_reranked_selected.csv"
COMPLETE_MARKER="$OUTPUT_DIR/COMPLETE"
mkdir -p "$OUTPUT_DIR/model_eval"

for required in "$CHECKPOINT" "$EVAL_CSV" "$TRAIN_FEATURES" "$EVAL_FEATURES"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: missing P1 input: $required" >&2
    exit 2
  fi
done

if [[ -f "$COMPLETE_MARKER" && -s "$CANDIDATE_CSV" ]]; then
  echo "P1 candidate pool already complete; reusing $CANDIDATE_CSV"
  exit 0
fi

echo "P1 candidate generation"
echo "  variant=$VARIANT"
echo "  seed=$SEED"
echo "  checkpoint=$CHECKPOINT"
echo "  eval_csv=$EVAL_CSV"
echo "  output=$CANDIDATE_CSV"
echo "  num_samples=$NUM_SAMPLES"
echo "  decoding=temperature:$TEMPERATURE top_k:$TOP_K top_p:$TOP_P repetition_penalty:$REPETITION_PENALTY"
echo "  note=diagnostic selected CSV is property-reranked and excluded from P1 metrics"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_direct_smiles_generator.py" \
  --eval-only \
  --eval-csv "$EVAL_CSV" \
  --resume-checkpoint "$CHECKPOINT" \
  --condition-features-dir "$TRAIN_FEATURES" \
  --eval-condition-features-dir "$EVAL_FEATURES" \
  --condition-mixing-mode append_property_program \
  --output-dir "$OUTPUT_DIR/model_eval" \
  --prediction-csv "$DIAGNOSTIC_SELECTED_CSV" \
  --candidate-output-csv "$CANDIDATE_CSV" \
  --eval-batch-size "${SUCC_P1_EVAL_BATCH_SIZE:-32}" \
  --max-new-tokens "${SUCC_P1_MAX_NEW_TOKENS:-96}" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --top-p "$TOP_P" \
  --num-samples "$NUM_SAMPLES" \
  --parallel-samples "${SUCC_P1_PARALLEL_SAMPLES:-8}" \
  --max-parallel-sequences "${SUCC_P1_MAX_PARALLEL_SEQUENCES:-1024}" \
  --repetition-penalty "$REPETITION_PENALTY" \
  --no-repeat-ngram-size "${SUCC_P1_NO_REPEAT_NGRAM_SIZE:-6}" \
  --min-new-tokens "${SUCC_P1_MIN_NEW_TOKENS:-6}" \
  --seed "$SEED" \
  --device "${SUCC_P1_DEVICE:-auto}"

cp "$SCRIPT_DIR/p1_property_program_group_rl_preregistration.json" "$OUTPUT_DIR/protocol_snapshot.json"
touch "$COMPLETE_MARKER"
echo "P1 variant complete: $VARIANT"
echo "candidate_csv=$CANDIDATE_CSV"
