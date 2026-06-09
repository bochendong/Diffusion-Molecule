#!/usr/bin/env bash
# Submit SUCC UniVideo MolEdit repair run: Table1-balanced, active-loss, weighted MW/SA/RB.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
export SUCC_DATASET_MODE=moledit
export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v2_fix}"
export SUCC_CONDITION_FEATURES_DIR="${SUCC_CONDITION_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_moledit_hf_vlm_v2_fix}"
export SUCC_HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
export SUCC_MOLEDIT_TRAIN_SPLIT="${SUCC_MOLEDIT_TRAIN_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
export SUCC_MOLEDIT_EVAL_SPLIT="${SUCC_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
export SUCC_MOLEDIT_TABLE1_TASKS_ONLY="${SUCC_MOLEDIT_TABLE1_TASKS_ONLY:-1}"
export SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK="${SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK:-5000}"
export SUCC_MOLEDIT_BALANCED_EVAL_PER_TASK="${SUCC_MOLEDIT_BALANCED_EVAL_PER_TASK:-100}"
export SUCC_MOLEDIT_TRAIN_LIMIT="${SUCC_MOLEDIT_TRAIN_LIMIT:-50000}"
export SUCC_MOLEDIT_EVAL_LIMIT="${SUCC_MOLEDIT_EVAL_LIMIT:-1000}"

export SUCC_MIN_EDIT_SOURCE_TANIMOTO="${SUCC_MIN_EDIT_SOURCE_TANIMOTO:-0.0}"
export SUCC_REQUIRE_EDIT_QUALITY_COLUMNS=0
export SUCC_REQUIRE_EVAL_ORACLE_STRICT=0

export SUCC_LATENT_BACKEND="${SUCC_LATENT_BACKEND:-image_vae}"
export SUCC_LATENT_TARGET_MODE="${SUCC_LATENT_TARGET_MODE:-residual}"
export SUCC_DIFFUSION_OBJECTIVE="${SUCC_DIFFUSION_OBJECTIVE:-pred_x0}"
export SUCC_RESIDUAL_SAMPLE_SCALE="${SUCC_RESIDUAL_SAMPLE_SCALE:-1.25}"
export SUCC_CONNECTOR_LATENT_BLEND="${SUCC_CONNECTOR_LATENT_BLEND:-0.10}"
export SUCC_SAMPLE_ETA="${SUCC_SAMPLE_ETA:-0.0}"
export SUCC_IMAGE_VAE_INK_LOSS_WEIGHT="${SUCC_IMAGE_VAE_INK_LOSS_WEIGHT:-4.0}"

export SUCC_STAGE1_EPOCHS="${SUCC_STAGE1_EPOCHS:-3}"
export SUCC_STAGE2_EPOCHS="${SUCC_STAGE2_EPOCHS:-8}"
export SUCC_STAGE3_EPOCHS="${SUCC_STAGE3_EPOCHS:-3}"
export SUCC_AUX_LOSS_WEIGHT="${SUCC_AUX_LOSS_WEIGHT:-0.50}"
export SUCC_SAMPLING_STRATEGY="${SUCC_SAMPLING_STRATEGY:-weighted}"
export SUCC_TABLE1_SAMPLE_WEIGHT="${SUCC_TABLE1_SAMPLE_WEIGHT:-1.0}"
export SUCC_TRAIN_PROPERTY_SAMPLE_WEIGHTS="${SUCC_TRAIN_PROPERTY_SAMPLE_WEIGHTS:-MW=4,SA=3,RB=2}"
export SUCC_AUX_PROPERTY_WEIGHTS="${SUCC_AUX_PROPERTY_WEIGHTS:-MW=3,SA=3,RB=2}"
export SUCC_AUX_ALL_PROPERTIES="${SUCC_AUX_ALL_PROPERTIES:-0}"

export SUCC_RUN_DATASET_EXPORT="${SUCC_RUN_DATASET_EXPORT:-1}"
export SUCC_RUN_FEATURE_EXPORT="${SUCC_RUN_FEATURE_EXPORT:-auto}"
export SUCC_DECODE_EVAL_IMAGES="${SUCC_DECODE_EVAL_IMAGES:-0}"
export SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=0
export SUCC_RUN_MOLSCRIBE_OCR=0
export SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER="${SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER:-1}"
export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-primary_fast}"
export SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY="${SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY:-skip-task}"

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

echo "Submitting SUCC UniVideo MolEdit v2 repair pipeline"
echo "  output_dir=$SUCC_UNIFIED_OUTPUT_DIR"
echo "  condition_features_dir=$SUCC_CONDITION_FEATURES_DIR"
echo "  table1_tasks_only=$SUCC_MOLEDIT_TABLE1_TASKS_ONLY"
echo "  balanced_train_per_task=$SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK"
echo "  balanced_eval_per_task=$SUCC_MOLEDIT_BALANCED_EVAL_PER_TASK"
echo "  sampling_strategy=$SUCC_SAMPLING_STRATEGY"
echo "  train_property_sample_weights=$SUCC_TRAIN_PROPERTY_SAMPLE_WEIGHTS"
echo "  aux_property_weights=$SUCC_AUX_PROPERTY_WEIGHTS"
echo "  aux_loss_weight=$SUCC_AUX_LOSS_WEIGHT"
echo "  residual_sample_scale=$SUCC_RESIDUAL_SAMPLE_SCALE"
echo "  connector_latent_blend=$SUCC_CONNECTOR_LATENT_BLEND"

bash "$SCRIPT_DIR/submit_univideo_molecule_pipeline.sh"
