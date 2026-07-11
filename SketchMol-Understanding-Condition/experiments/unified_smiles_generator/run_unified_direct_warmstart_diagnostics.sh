#!/usr/bin/env bash
# Small warm-start beam diagnostics for the unified direct-checkpoint rescue line.
#
# Profiles (SUCC_UNIFIED_WARMSTART_DIAG_PROFILE):
#   direct_compat_with_image   - suite v1 features, direct_compat, with_image (default)
#   direct_compat_text_only    - suite v1 features, direct_compat, text_only
#   property_program_only      - no frozen VLM features, property-program tokens only
#   direct_feature_store       - direct 2p7p feature dirs + direct_compat + with_image

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PROFILE="${SUCC_UNIFIED_WARMSTART_DIAG_PROFILE:-direct_compat_with_image}"
DIAG_ROOT="${SUCC_UNIFIED_DIRECT_WARMSTART_OUTPUT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_direct_warmstart_diagnostics_v2}"
DIRECT_FEATURE_ROOT="${SUCC_UNIFIED_DIRECT_WARMSTART_DIRECT_FEATURE_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
PHASE0_SAMPLE="${SUCC_UNIFIED_PHASE0_DIRECT_SAMPLE:-1}"
PHASE="${SUCC_UNIFIED_DIRECT_COMPAT_PHASE:-phase0}"

run_profile() {
  local profile="$1"
  local output_root="$DIAG_ROOT/$profile"

  SOURCE_SUITE_ROOT="${SUCC_UNIFIED_DIRECT_WARMSTART_SOURCE_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"

  export SUCC_UNIFIED_DIRECT_WARMSTART_OUTPUT_ROOT="$output_root"
  export SUCC_UNIFIED_DIRECT_WARMSTART_SOURCE_SUITE_ROOT="$SOURCE_SUITE_ROOT"
  export SUCC_UNIFIED_DIRECT_WARMSTART_RUN_GRPO="${SUCC_UNIFIED_DIRECT_WARMSTART_RUN_GRPO:-0}"
  export SUCC_UNIFIED_DIRECT_WARMSTART_RUN_BASELINE="${SUCC_UNIFIED_DIRECT_WARMSTART_RUN_BASELINE:-1}"
  export SUCC_UNIFIED_EVAL_LIMIT="${SUCC_UNIFIED_EVAL_LIMIT:-500}"
  export SUCC_UNIFIED_DIRECT_WARMSTART_BENCHMARK_TASKS="${SUCC_UNIFIED_DIRECT_WARMSTART_BENCHMARK_TASKS:-denovo_2p7p}"

  if [[ "$PHASE0_SAMPLE" == "1" ]]; then
    if [[ "$PHASE" == "phase1" ]]; then
      default_num_samples=128
      default_top_k_candidates=128
      default_parallel=8
    else
      default_num_samples=32
      default_top_k_candidates=32
      default_parallel=4
    fi
    export SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_SIZE="${SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_SIZE:-20}"
    export SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_EXPAND_SIZE="${SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_EXPAND_SIZE:-32}"
    export SUCC_UNIFIED_DIRECT_WARMSTART_TOP_K_CANDIDATES="${SUCC_UNIFIED_DIRECT_WARMSTART_TOP_K_CANDIDATES:-$default_top_k_candidates}"
    export SUCC_UNIFIED_DECODING_MODE="${SUCC_UNIFIED_DECODING_MODE:-sample}"
    export SUCC_UNIFIED_NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-$default_num_samples}"
    export SUCC_UNIFIED_TEMPERATURE="${SUCC_UNIFIED_TEMPERATURE:-0.85}"
    export SUCC_UNIFIED_TOP_K="${SUCC_UNIFIED_TOP_K:-40}"
    export SUCC_UNIFIED_TOP_P="${SUCC_UNIFIED_TOP_P:-0.95}"
    export SUCC_UNIFIED_PARALLEL_SAMPLES="${SUCC_UNIFIED_PARALLEL_SAMPLES:-$default_parallel}"
  else
    export SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_SIZE="${SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_SIZE:-20}"
    export SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_EXPAND_SIZE="${SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_EXPAND_SIZE:-32}"
    export SUCC_UNIFIED_DIRECT_WARMSTART_TOP_K_CANDIDATES="${SUCC_UNIFIED_DIRECT_WARMSTART_TOP_K_CANDIDATES:-20}"
    export SUCC_UNIFIED_DECODING_MODE="${SUCC_UNIFIED_DECODING_MODE:-beam}"
    export SUCC_UNIFIED_NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-1}"
  fi

  unset SUCC_UNIFIED_DIRECT_WARMSTART_TRAIN_FEATURES_DIR || true
  unset SUCC_UNIFIED_DIRECT_WARMSTART_EVAL_FEATURES_DIR || true

  case "$profile" in
  direct_compat_with_image)
    export SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_LAYOUT="direct_compat"
    export SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_FEATURE_VARIANT="full"
    export SUCC_UNIFIED_DIRECT_WARMSTART_INPUT_MODALITY="with_image"
    export SUCC_UNIFIED_DIRECT_WARMSTART_TRAIN_FEATURES_DIR="$DIRECT_FEATURE_ROOT/train_condition_features_hf_vlm"
    export SUCC_UNIFIED_DIRECT_WARMSTART_EVAL_FEATURES_DIR="$DIRECT_FEATURE_ROOT/eval_condition_features_hf_vlm"
    ;;
  direct_compat_text_only)
    export SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_LAYOUT="direct_compat"
    export SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_FEATURE_VARIANT="full"
    export SUCC_UNIFIED_DIRECT_WARMSTART_INPUT_MODALITY="text_only"
    export SUCC_UNIFIED_DIRECT_WARMSTART_TRAIN_FEATURES_DIR="$DIRECT_FEATURE_ROOT/train_condition_features_hf_vlm"
    export SUCC_UNIFIED_DIRECT_WARMSTART_EVAL_FEATURES_DIR="$DIRECT_FEATURE_ROOT/eval_condition_features_hf_vlm"
    ;;
    property_program_only)
      export SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_LAYOUT="property_program_only"
      export SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_FEATURE_VARIANT="full"
      export SUCC_UNIFIED_DIRECT_WARMSTART_INPUT_MODALITY="with_image"
      ;;
    direct_feature_store)
      export SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_LAYOUT="direct_compat"
      export SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_FEATURE_VARIANT="full"
      export SUCC_UNIFIED_DIRECT_WARMSTART_INPUT_MODALITY="with_image"
      export SUCC_UNIFIED_DIRECT_WARMSTART_TRAIN_FEATURES_DIR="$DIRECT_FEATURE_ROOT/train_condition_features_hf_vlm"
      export SUCC_UNIFIED_DIRECT_WARMSTART_EVAL_FEATURES_DIR="$DIRECT_FEATURE_ROOT/eval_condition_features_hf_vlm"
      ;;
    *)
      echo "ERROR: unknown profile=$profile" >&2
      echo "Supported: direct_compat_with_image, direct_compat_text_only, property_program_only, direct_feature_store, all" >&2
      exit 2
      ;;
  esac

  echo "Unified direct warm-start diagnostic profile=$profile"
  echo "  output_root=$output_root"
  echo "  eval_limit=$SUCC_UNIFIED_EVAL_LIMIT"
  echo "  beam_size=$SUCC_UNIFIED_DIRECT_WARMSTART_BEAM_SIZE"
  echo "  decoding_mode=${SUCC_UNIFIED_DECODING_MODE:-beam}"
  echo "  num_samples=${SUCC_UNIFIED_NUM_SAMPLES:-1}"
  echo "  condition_layout=${SUCC_UNIFIED_DIRECT_WARMSTART_CONDITION_LAYOUT}"
  echo "  input_modality=${SUCC_UNIFIED_DIRECT_WARMSTART_INPUT_MODALITY:-default}"
  echo "  train_features_dir=${SUCC_UNIFIED_DIRECT_WARMSTART_TRAIN_FEATURES_DIR:-none}"
  echo "  eval_features_dir=${SUCC_UNIFIED_DIRECT_WARMSTART_EVAL_FEATURES_DIR:-none}"

  bash "$SCRIPT_DIR/run_unified_smiles_generator_direct_warmstart_grpo_beam.sh"
}

if [[ "$PROFILE" == "all" ]]; then
  for profile in direct_compat_with_image direct_compat_text_only property_program_only direct_feature_store; do
    echo
    echo "========================================"
    echo "Running diagnostic profile: $profile"
    echo "========================================"
    run_profile "$profile"
  done
  exit 0
fi

run_profile "$PROFILE"
