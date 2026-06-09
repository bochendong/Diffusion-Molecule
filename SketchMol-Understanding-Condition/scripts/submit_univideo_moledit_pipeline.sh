#!/usr/bin/env bash
# Submit SUCC UniVideo-style training against MolEdit-Instruct enhanced splits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
export SUCC_DATASET_MODE=moledit
export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v1}"
export SUCC_CONDITION_FEATURES_DIR="${SUCC_CONDITION_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_moledit_hf_vlm_v1}"
export SUCC_HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
export SUCC_MOLEDIT_TRAIN_SPLIT="${SUCC_MOLEDIT_TRAIN_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
export SUCC_MOLEDIT_EVAL_SPLIT="${SUCC_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
export SUCC_MIN_EDIT_SOURCE_TANIMOTO="${SUCC_MIN_EDIT_SOURCE_TANIMOTO:-0.0}"
export SUCC_REQUIRE_EDIT_QUALITY_COLUMNS=0
export SUCC_REQUIRE_EVAL_ORACLE_STRICT=0

export SUCC_LATENT_BACKEND="${SUCC_LATENT_BACKEND:-image_vae}"
export SUCC_LATENT_TARGET_MODE="${SUCC_LATENT_TARGET_MODE:-residual}"
export SUCC_DIFFUSION_OBJECTIVE="${SUCC_DIFFUSION_OBJECTIVE:-pred_x0}"
export SUCC_SAMPLE_ETA="${SUCC_SAMPLE_ETA:-0.0}"
export SUCC_IMAGE_VAE_INK_LOSS_WEIGHT="${SUCC_IMAGE_VAE_INK_LOSS_WEIGHT:-4.0}"

export SUCC_EVAL_LIMIT="${SUCC_EVAL_LIMIT:-1000}"
export SUCC_MOLEDIT_EVAL_LIMIT="${SUCC_MOLEDIT_EVAL_LIMIT:-$SUCC_EVAL_LIMIT}"
export SUCC_RUN_DATASET_EXPORT="${SUCC_RUN_DATASET_EXPORT:-1}"
export SUCC_RUN_FEATURE_EXPORT="${SUCC_RUN_FEATURE_EXPORT:-auto}"
export SUCC_DECODE_EVAL_IMAGES="${SUCC_DECODE_EVAL_IMAGES:-0}"
export SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=0
export SUCC_RUN_MOLSCRIBE_OCR=0
export SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER="${SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER:-1}"
export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-primary_fast}"

# Reuse the validated ink-aware image VAE unless the caller asks to retrain.
if [[ -z "${SUCC_IMAGE_VAE_CHECKPOINT:-}" ]]; then
  CANONICAL_VAE="SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/molecule_image_vae/molecule_image_vae.pt"
  if [[ -f "$CANONICAL_VAE" ]]; then
    export SUCC_IMAGE_VAE_CHECKPOINT="$CANONICAL_VAE"
    export SUCC_IMAGE_VAE_DIR="$(dirname "$CANONICAL_VAE")"
    export SUCC_RUN_IMAGE_VAE_TRAIN="${SUCC_RUN_IMAGE_VAE_TRAIN:-0}"
  fi
fi
export SUCC_RUN_IMAGE_VAE_TRAIN="${SUCC_RUN_IMAGE_VAE_TRAIN:-auto}"

echo "Submitting SUCC UniVideo MolEdit-Instruct pipeline"
echo "  output_dir=$SUCC_UNIFIED_OUTPUT_DIR"
echo "  condition_features_dir=$SUCC_CONDITION_FEATURES_DIR"
echo "  moledit_train_split=$SUCC_MOLEDIT_TRAIN_SPLIT"
echo "  moledit_eval_split=$SUCC_MOLEDIT_EVAL_SPLIT"
echo "  latent_backend=$SUCC_LATENT_BACKEND"
echo "  latent_target_mode=$SUCC_LATENT_TARGET_MODE"
echo "  diffusion_objective=$SUCC_DIFFUSION_OBJECTIVE"
echo "  decode_eval_images=$SUCC_DECODE_EVAL_IMAGES"
echo "  run_molscribe_ocr=$SUCC_RUN_MOLSCRIBE_OCR"
echo "  submit_materialized_after=$SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER"

bash "$SCRIPT_DIR/submit_univideo_molecule_pipeline.sh"
