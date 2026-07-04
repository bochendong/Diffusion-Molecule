#!/usr/bin/env bash
# Run unified-generator experiments across decoding and input-modality lines.
#
# Default behavior is inference/benchmark-only from existing checkpoints. Set
# SUCC_UNIFIED_SUITE_RUN_TRAIN=1 and/or SUCC_UNIFIED_SUITE_RUN_RL=1 to run the
# heavier training stages before benchmarks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SUITE_ROOT="${SUCC_UNIFIED_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
MODALITIES="${SUCC_UNIFIED_SUITE_MODALITIES:-with_image,no_image}"
DECODING_MODES="${SUCC_UNIFIED_SUITE_DECODING_MODES:-sample,beam}"
RUN_TRAIN="${SUCC_UNIFIED_SUITE_RUN_TRAIN:-0}"
RUN_RL="${SUCC_UNIFIED_SUITE_RUN_RL:-0}"
RUN_BENCHMARK="${SUCC_UNIFIED_SUITE_RUN_BENCHMARK:-1}"
RUN_FEATURE_EXPORT="${SUCC_UNIFIED_SUITE_RUN_FEATURE_EXPORT:-0}"
TRAIN_CSV="${SUCC_UNIFIED_TRAIN_CSV:-}"
EVAL_CSV="${SUCC_UNIFIED_EVAL_CSV:-}"
FEATURE_VARIANTS_OUTPUT_ROOT="${SUCC_UNIFIED_FEATURE_VARIANTS_OUTPUT_ROOT:-$SUITE_ROOT/feature_variants}"
DEFAULT_TRAIN_FEATURES_DIR="${SUCC_UNIFIED_TRAIN_FEATURES_DIR:-$FEATURE_VARIANTS_OUTPUT_ROOT/train_condition_features_hf_vlm}"
DEFAULT_EVAL_FEATURES_DIR="${SUCC_UNIFIED_EVAL_FEATURES_DIR:-$FEATURE_VARIANTS_OUTPUT_ROOT/eval_condition_features_hf_vlm}"
BENCHMARK_TASKS="${SUCC_UNIFIED_BENCHMARK_TASKS:-denovo_2p7p,external_multiproperty,moledit_table1}"
MOLEDIT_REFERENCE_CSV="${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV:-}"
MOLEDIT_BUDGETS="${SUCC_UNIFIED_MOLEDIT_BUDGETS:-20,256}"
EXTERNAL_GENERATED_PROPERTIES_CSV="${SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV:-}"
EXTERNAL_SOURCE_PROPERTIES_CSV="${SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV:-}"
SAMPLE_NUM_SAMPLES="${SUCC_UNIFIED_SUITE_SAMPLE_NUM_SAMPLES:-40}"
BEAM_SIZE="${SUCC_UNIFIED_SUITE_BEAM_SIZE:-40}"
TOP_K_CANDIDATES="${SUCC_UNIFIED_TOP_K_CANDIDATES:-40}"
SEED="${SUCC_UNIFIED_SEED:-7}"
DEVICE="${SUCC_DEVICE:-${SUCC_UNIFIED_DEVICE:-auto}}"

mkdir -p "$SUITE_ROOT"

if [[ "$RUN_FEATURE_EXPORT" == "1" ]]; then
  SUCC_UNIFIED_FEATURE_VARIANTS_OUTPUT_ROOT="$FEATURE_VARIANTS_OUTPUT_ROOT" \
  SUCC_UNIFIED_FEATURE_VARIANTS="${SUCC_UNIFIED_FEATURE_VARIANTS:-full,text_only}" \
  SUCC_UNIFIED_TRAIN_CSV="$TRAIN_CSV" \
  SUCC_UNIFIED_EVAL_CSV="$EVAL_CSV" \
  bash "$SCRIPT_DIR/run_unified_smiles_generator_feature_variants.sh"
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  echo "$value"
}

variant_for_modality() {
  case "$1" in
    with_image) echo "full" ;;
    no_image) echo "text_only" ;;
    image_only) echo "image_only" ;;
    *) echo "$1" ;;
  esac
}

checkpoint_for_modality() {
  local modality="$1"
  local modality_upper
  modality_upper="$(echo "$modality" | tr '[:lower:]' '[:upper:]' | tr '-' '_' | tr ' ' '_')"
  local env_name="SUCC_UNIFIED_${modality_upper}_CHECKPOINT"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    echo "$value"
  else
    echo "$SUITE_ROOT/$modality/group_rl/unified_smiles_generator_group_rl.pt"
  fi
}

run_modality_train() {
  local modality="$1"
  local variant="$2"
  local modality_dir="$SUITE_ROOT/$modality"
  if [[ -z "$TRAIN_CSV" ]]; then
    echo "ERROR: SUCC_UNIFIED_TRAIN_CSV is required for SUCC_UNIFIED_SUITE_RUN_TRAIN=1" >&2
    exit 2
  fi
  SUCC_UNIFIED_OUTPUT_DIR="$modality_dir/sft" \
  SUCC_UNIFIED_TRAIN_CSV="$TRAIN_CSV" \
  SUCC_UNIFIED_EVAL_CSV="$EVAL_CSV" \
  SUCC_UNIFIED_TRAIN_FEATURES_DIR="$DEFAULT_TRAIN_FEATURES_DIR" \
  SUCC_UNIFIED_EVAL_FEATURES_DIR="$DEFAULT_EVAL_FEATURES_DIR" \
  SUCC_UNIFIED_CONDITION_FEATURE_VARIANT="$variant" \
  SUCC_UNIFIED_INPUT_MODALITY="$modality" \
  SUCC_UNIFIED_SEED="$SEED" \
  SUCC_UNIFIED_DEVICE="$DEVICE" \
  bash "$SCRIPT_DIR/run_unified_smiles_generator_train.sh"
}

run_modality_rl() {
  local modality="$1"
  local variant="$2"
  local modality_dir="$SUITE_ROOT/$modality"
  local resume="${SUCC_UNIFIED_RL_RESUME_CHECKPOINT:-$modality_dir/sft/unified_smiles_generator.pt}"
  if [[ -z "$TRAIN_CSV" ]]; then
    echo "ERROR: SUCC_UNIFIED_TRAIN_CSV is required for SUCC_UNIFIED_SUITE_RUN_RL=1" >&2
    exit 2
  fi
  SUCC_UNIFIED_RL_OUTPUT_DIR="$modality_dir/group_rl" \
  SUCC_UNIFIED_RL_RESUME_CHECKPOINT="$resume" \
  SUCC_UNIFIED_RL_TRAIN_CSV="$TRAIN_CSV" \
  SUCC_UNIFIED_RL_EVAL_CSV="$EVAL_CSV" \
  SUCC_UNIFIED_RL_TRAIN_FEATURES_DIR="$DEFAULT_TRAIN_FEATURES_DIR" \
  SUCC_UNIFIED_RL_EVAL_FEATURES_DIR="$DEFAULT_EVAL_FEATURES_DIR" \
  SUCC_UNIFIED_CONDITION_FEATURE_VARIANT="$variant" \
  SUCC_UNIFIED_INPUT_MODALITY="$modality" \
  SUCC_UNIFIED_RL_SEED="$SEED" \
  SUCC_UNIFIED_DEVICE="$DEVICE" \
  bash "$SCRIPT_DIR/run_unified_smiles_generator_group_rl.sh"
}

run_modality_benchmark() {
  local modality="$1"
  local variant="$2"
  local decoding="$3"
  local modality_dir="$SUITE_ROOT/$modality"
  local checkpoint
  checkpoint="$(checkpoint_for_modality "$modality")"
  local run_dir="$modality_dir/benchmark_${decoding}"
  local num_samples="$SAMPLE_NUM_SAMPLES"
  local beam_size="$BEAM_SIZE"
  if [[ "$decoding" == "sample" ]]; then
    beam_size="${SUCC_UNIFIED_SUITE_SAMPLE_BEAM_SIZE:-1}"
  elif [[ "$decoding" == "beam" ]]; then
    num_samples="${SUCC_UNIFIED_SUITE_BEAM_NUM_SAMPLES:-1}"
  else
    echo "ERROR: unsupported decoding mode in suite: $decoding" >&2
    exit 2
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: missing checkpoint for $modality: $checkpoint" >&2
    echo "       Set SUCC_UNIFIED_${modality^^}_CHECKPOINT or enable training/RL stages." >&2
    exit 2
  fi
  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1 \
  SUCC_UNIFIED_CHECKPOINT="$checkpoint" \
  SUCC_UNIFIED_EVAL_CSV="$EVAL_CSV" \
  SUCC_UNIFIED_EVAL_FEATURES_DIR="$DEFAULT_EVAL_FEATURES_DIR" \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$run_dir" \
  SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="$run_dir/sample_outputs" \
  SUCC_UNIFIED_BENCHMARK_TASKS="$BENCHMARK_TASKS" \
  SUCC_UNIFIED_CONDITION_FEATURE_VARIANT="$variant" \
  SUCC_UNIFIED_INPUT_MODALITY="$modality" \
  SUCC_UNIFIED_METHOD_NAME="unified_smiles_generator_${modality}_${decoding}" \
  SUCC_UNIFIED_DECODING_MODE="$decoding" \
  SUCC_UNIFIED_NUM_SAMPLES="$num_samples" \
  SUCC_UNIFIED_BEAM_SIZE="$beam_size" \
  SUCC_UNIFIED_TOP_K_CANDIDATES="$TOP_K_CANDIDATES" \
  SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$MOLEDIT_REFERENCE_CSV" \
  SUCC_UNIFIED_MOLEDIT_BUDGETS="$MOLEDIT_BUDGETS" \
  SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV="$EXTERNAL_GENERATED_PROPERTIES_CSV" \
  SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV="$EXTERNAL_SOURCE_PROPERTIES_CSV" \
  SUCC_UNIFIED_SEED="$SEED" \
  SUCC_UNIFIED_DEVICE="$DEVICE" \
  bash "$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh"
}

echo "Unified SMILES generator experiment suite"
echo "  suite_root=$SUITE_ROOT"
echo "  modalities=$MODALITIES"
echo "  decoding_modes=$DECODING_MODES"
echo "  run_feature_export=$RUN_FEATURE_EXPORT"
echo "  run_train=$RUN_TRAIN"
echo "  run_rl=$RUN_RL"
echo "  run_benchmark=$RUN_BENCHMARK"
echo "  benchmark_tasks=$BENCHMARK_TASKS"
echo "  sample_num_samples=$SAMPLE_NUM_SAMPLES"
echo "  beam_size=$BEAM_SIZE"

IFS=',' read -r -a modality_array <<< "$MODALITIES"
IFS=',' read -r -a decoding_array <<< "$DECODING_MODES"
for raw_modality in "${modality_array[@]}"; do
  modality="$(trim "$raw_modality")"
  [[ -z "$modality" ]] && continue
  variant="$(variant_for_modality "$modality")"
  echo
  echo "=== modality=$modality variant=$variant ==="
  if [[ "$RUN_TRAIN" == "1" ]]; then
    run_modality_train "$modality" "$variant"
  fi
  if [[ "$RUN_RL" == "1" ]]; then
    run_modality_rl "$modality" "$variant"
  fi
  if [[ "$RUN_BENCHMARK" == "1" ]]; then
    for raw_decoding in "${decoding_array[@]}"; do
      decoding="$(trim "$raw_decoding")"
      [[ -z "$decoding" ]] && continue
      echo
      echo "--- benchmark modality=$modality decoding=$decoding ---"
      run_modality_benchmark "$modality" "$variant" "$decoding"
    done
  fi
done

echo
echo "Unified experiment suite complete."
echo "  suite_root=$SUITE_ROOT"
