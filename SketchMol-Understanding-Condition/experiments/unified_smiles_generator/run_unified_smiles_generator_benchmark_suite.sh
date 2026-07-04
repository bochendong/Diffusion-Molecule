#!/usr/bin/env bash
# Run existing benchmark evaluators on unified SMILES selected/candidate CSVs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
OUTPUT_DIR="${SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_benchmark_suite}"
TASKS="${SUCC_UNIFIED_BENCHMARK_TASKS:-denovo_2p7p,external_multiproperty,moledit_table1}"
RUN_SAMPLE="${SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE:-0}"

CHECKPOINT="${SUCC_UNIFIED_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_group_rl_v1/unified_smiles_generator_group_rl.pt}"
EVAL_CSV="${SUCC_UNIFIED_EVAL_CSV:-}"
EVAL_FEATURES_DIR="${SUCC_UNIFIED_EVAL_FEATURES_DIR:-${SUCC_UNIFIED_FEATURES_DIR:-}}"
SAMPLE_OUTPUT_DIR="${SUCC_UNIFIED_SAMPLE_OUTPUT_DIR:-$OUTPUT_DIR/sample}"
PREDICTION_CSV="${SUCC_UNIFIED_BENCHMARK_PREDICTION_CSV:-$SAMPLE_OUTPUT_DIR/unified_smiles_predictions.csv}"
CANDIDATE_CSV="${SUCC_UNIFIED_BENCHMARK_CANDIDATE_CSV:-$SAMPLE_OUTPUT_DIR/unified_smiles_candidate_predictions.csv}"

CONDITION_FEATURE_ARRAY="${SUCC_UNIFIED_CONDITION_FEATURE_ARRAY:-query_tokens}"
CONDITION_FEATURE_VARIANT="${SUCC_UNIFIED_CONDITION_FEATURE_VARIANT:-full}"
INPUT_MODALITY="${SUCC_UNIFIED_INPUT_MODALITY:-}"
CONDITION_DIM="${SUCC_UNIFIED_CONDITION_DIM:-256}"
MAX_SMILES_LENGTH="${SUCC_UNIFIED_MAX_SMILES_LENGTH:-160}"
MAX_SOURCE_TOKENS="${SUCC_UNIFIED_MAX_SOURCE_TOKENS:-96}"
DECODING_MODE="${SUCC_UNIFIED_DECODING_MODE:-sample}"
NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-40}"
BEAM_SIZE="${SUCC_UNIFIED_BEAM_SIZE:-40}"
BEAM_EXPAND_SIZE="${SUCC_UNIFIED_BEAM_EXPAND_SIZE:-64}"
BEAM_LENGTH_PENALTY="${SUCC_UNIFIED_BEAM_LENGTH_PENALTY:-0.8}"
TOP_K_CANDIDATES="${SUCC_UNIFIED_TOP_K_CANDIDATES:-40}"
TEMPERATURE="${SUCC_UNIFIED_TEMPERATURE:-0.7}"
TOP_K="${SUCC_UNIFIED_TOP_K:-24}"
TOP_P="${SUCC_UNIFIED_TOP_P:-0.9}"
SEED="${SUCC_UNIFIED_SEED:-7}"
DEVICE="${SUCC_DEVICE:-${SUCC_UNIFIED_DEVICE:-auto}}"

if [[ -z "$INPUT_MODALITY" ]]; then
  case "$CONDITION_FEATURE_VARIANT" in
    full|image_only) INPUT_MODALITY="with_image" ;;
    text_only|caption_bottleneck) INPUT_MODALITY="no_image" ;;
    random_query) INPUT_MODALITY="random_query" ;;
    *) INPUT_MODALITY="$CONDITION_FEATURE_VARIANT" ;;
  esac
fi
METHOD_NAME="${SUCC_UNIFIED_METHOD_NAME:-unified_smiles_generator_${INPUT_MODALITY}}"

EXTERNAL_GENERATED_PROPERTIES_CSV="${SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV:-}"
EXTERNAL_SOURCE_PROPERTIES_CSV="${SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV:-}"
EXTERNAL_GROUP_COLUMN="${SUCC_UNIFIED_EXTERNAL_GROUP_COLUMN:-condition_id}"
EXTERNAL_MIN_SOURCE_TANIMOTO="${SUCC_UNIFIED_EXTERNAL_MIN_SOURCE_TANIMOTO:-0.4}"
MOLEDIT_REFERENCE_CSV="${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV:-}"
MOLEDIT_BUDGETS="${SUCC_UNIFIED_MOLEDIT_BUDGETS:-20,256}"
MOLEDIT_SELECTION_MODE="${SUCC_UNIFIED_MOLEDIT_SELECTION_MODE:-unified_score}"
MOLEDIT_THRESHOLDS="${SUCC_UNIFIED_MOLEDIT_THRESHOLDS:-0.65,0.15}"
MOLEDIT_MISSING_ORACLE_POLICY="${SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY:-fail}"
MOLEDIT_REQUIRE_TABLE1_COVERAGE="${SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE:-0}"

mkdir -p "$OUTPUT_DIR" "$SAMPLE_OUTPUT_DIR"

if [[ "$RUN_SAMPLE" == "1" ]]; then
  if [[ -z "$EVAL_CSV" ]]; then
    echo "ERROR: SUCC_UNIFIED_EVAL_CSV is required when SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1" >&2
    exit 2
  fi
  sample_args=(
    "$SCRIPT_DIR/unified_smiles_generator.py"
    sample
    --checkpoint "$CHECKPOINT"
    --eval-csv "$EVAL_CSV"
    --output-dir "$SAMPLE_OUTPUT_DIR"
    --prediction-csv "$PREDICTION_CSV"
    --candidate-output-csv "$CANDIDATE_CSV"
    --condition-feature-array "$CONDITION_FEATURE_ARRAY"
    --condition-feature-variant "$CONDITION_FEATURE_VARIANT"
    --condition-dim "$CONDITION_DIM"
    --max-smiles-length "$MAX_SMILES_LENGTH"
    --max-source-tokens "$MAX_SOURCE_TOKENS"
    --decoding-mode "$DECODING_MODE"
    --num-samples "$NUM_SAMPLES"
    --beam-size "$BEAM_SIZE"
    --beam-expand-size "$BEAM_EXPAND_SIZE"
    --beam-length-penalty "$BEAM_LENGTH_PENALTY"
    --top-k-candidates "$TOP_K_CANDIDATES"
    --temperature "$TEMPERATURE"
    --top-k "$TOP_K"
    --top-p "$TOP_P"
    --seed "$SEED"
    --device "$DEVICE"
  )
  if [[ -n "$INPUT_MODALITY" ]]; then
    sample_args+=(--input-modality "$INPUT_MODALITY")
  fi
  if [[ -n "$EVAL_FEATURES_DIR" ]]; then
    sample_args+=(--eval-condition-features-dir "$EVAL_FEATURES_DIR")
  fi
  if [[ -n "${SUCC_UNIFIED_EVAL_LIMIT:-}" ]]; then
    sample_args+=(--eval-limit "$SUCC_UNIFIED_EVAL_LIMIT")
  fi
  echo "Unified benchmark suite: sampling first"
  echo "  checkpoint=$CHECKPOINT"
  echo "  eval_csv=$EVAL_CSV"
  echo "  prediction_csv=$PREDICTION_CSV"
  echo "  candidate_csv=$CANDIDATE_CSV"
  "$PYTHON_BIN" "${sample_args[@]}"
fi

runner_args=(
  "$SCRIPT_DIR/unified_benchmark_runner.py"
  --prediction-csv "$PREDICTION_CSV"
  --candidate-prediction-csv "$CANDIDATE_CSV"
  --output-dir "$OUTPUT_DIR"
  --tasks "$TASKS"
  --python-bin "$PYTHON_BIN"
  --method-name "$METHOD_NAME"
  --external-group-column "$EXTERNAL_GROUP_COLUMN"
  --external-min-source-tanimoto "$EXTERNAL_MIN_SOURCE_TANIMOTO"
  --moledit-budgets "$MOLEDIT_BUDGETS"
  --moledit-selection-mode "$MOLEDIT_SELECTION_MODE"
  --moledit-thresholds "$MOLEDIT_THRESHOLDS"
  --moledit-missing-oracle-policy "$MOLEDIT_MISSING_ORACLE_POLICY"
)

if [[ -n "$EXTERNAL_GENERATED_PROPERTIES_CSV" ]]; then
  runner_args+=(--external-generated-properties-csv "$EXTERNAL_GENERATED_PROPERTIES_CSV")
fi
if [[ -n "$EXTERNAL_SOURCE_PROPERTIES_CSV" ]]; then
  runner_args+=(--external-source-properties-csv "$EXTERNAL_SOURCE_PROPERTIES_CSV")
fi
if [[ -n "$MOLEDIT_REFERENCE_CSV" ]]; then
  runner_args+=(--moledit-reference-csv "$MOLEDIT_REFERENCE_CSV")
fi
if [[ "$MOLEDIT_REQUIRE_TABLE1_COVERAGE" == "1" ]]; then
  runner_args+=(--moledit-require-table1-coverage)
fi

echo "Unified benchmark suite"
echo "  tasks=$TASKS"
echo "  prediction_csv=$PREDICTION_CSV"
echo "  candidate_csv=$CANDIDATE_CSV"
echo "  condition_feature_variant=$CONDITION_FEATURE_VARIANT"
echo "  input_modality=${INPUT_MODALITY:-auto}"
echo "  output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" "${runner_args[@]}"
