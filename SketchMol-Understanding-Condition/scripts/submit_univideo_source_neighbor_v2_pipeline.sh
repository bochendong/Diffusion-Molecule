#!/usr/bin/env bash
# Submit the source-neighbor UniVideo v2-style retraining pipeline.
#
# This launcher keeps the stable v2 choices (image_vae + residual target +
# pred_x0 + ink-aware VAE) but points the data path at the source-neighbor
# multi-property dataset and schedules the OCR-free materialized benchmark after
# training succeeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=./multiproperty_dataset_defaults.sh
source "$SCRIPT_DIR/multiproperty_dataset_defaults.sh"
export_smmed_source_neighbor_defaults
export_succ_edit_quality_defaults

export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-$SUCC_DEFAULT_UNIVIDEO_OUTPUT_DIR}"
export SUCC_CONDITION_ROWS="${SUCC_CONDITION_ROWS:-$SMMED_DEFAULT_CONDITION_ROWS}"
export SUCC_CONDITION_FEATURES_DIR="${SUCC_CONDITION_FEATURES_DIR:-$SUCC_DEFAULT_FEATURES_DIR}"
export SUCC_HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"

export SUCC_LATENT_BACKEND="${SUCC_LATENT_BACKEND:-image_vae}"
export SUCC_LATENT_TARGET_MODE="${SUCC_LATENT_TARGET_MODE:-residual}"
export SUCC_DIFFUSION_OBJECTIVE="${SUCC_DIFFUSION_OBJECTIVE:-pred_x0}"
export SUCC_SAMPLE_ETA="${SUCC_SAMPLE_ETA:-0.0}"
export SUCC_IMAGE_VAE_INK_LOSS_WEIGHT="${SUCC_IMAGE_VAE_INK_LOSS_WEIGHT:-4.0}"

export SUCC_EVAL_LIMIT="${SUCC_EVAL_LIMIT:-1000}"
export SUCC_MAX_DECODE_IMAGES="${SUCC_MAX_DECODE_IMAGES:-$SUCC_EVAL_LIMIT}"
export SUCC_RUN_DATASET_EXPORT="${SUCC_RUN_DATASET_EXPORT:-1}"
export SUCC_RUN_FEATURE_EXPORT="${SUCC_RUN_FEATURE_EXPORT:-auto}"
export SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK="${SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK:-prepare}"
export SUCC_RUN_MOLSCRIBE_OCR="${SUCC_RUN_MOLSCRIBE_OCR:-0}"
export SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER="${SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER:-1}"
export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-primary_fast}"

if [[ -z "${SUCC_IMAGE_VAE_CHECKPOINT:-}" && -f "$SUCC_CANONICAL_V2_IMAGE_VAE_CHECKPOINT" ]]; then
  export SUCC_IMAGE_VAE_CHECKPOINT="$SUCC_CANONICAL_V2_IMAGE_VAE_CHECKPOINT"
  export SUCC_IMAGE_VAE_DIR="$(dirname "$SUCC_CANONICAL_V2_IMAGE_VAE_CHECKPOINT")"
  export SUCC_RUN_IMAGE_VAE_TRAIN="${SUCC_RUN_IMAGE_VAE_TRAIN:-0}"
else
  export SUCC_RUN_IMAGE_VAE_TRAIN="${SUCC_RUN_IMAGE_VAE_TRAIN:-auto}"
fi

echo "Submitting source-neighbor UniVideo v2-style retraining"
echo "  output_dir=$SUCC_UNIFIED_OUTPUT_DIR"
echo "  condition_rows=$SUCC_CONDITION_ROWS"
echo "  condition_features_dir=$SUCC_CONDITION_FEATURES_DIR"
echo "  hf_model=$SUCC_HF_MODEL_NAME_OR_PATH"
echo "  latent_backend=$SUCC_LATENT_BACKEND"
echo "  latent_target_mode=$SUCC_LATENT_TARGET_MODE"
echo "  diffusion_objective=$SUCC_DIFFUSION_OBJECTIVE"
echo "  image_vae_checkpoint=${SUCC_IMAGE_VAE_CHECKPOINT:-auto-train}"
echo "  run_image_structure_benchmark=$SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK"
echo "  submit_materialized_after=$SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER"

bash "$SCRIPT_DIR/submit_univideo_molecule_pipeline.sh"
