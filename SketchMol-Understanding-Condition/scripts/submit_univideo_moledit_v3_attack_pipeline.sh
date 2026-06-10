#!/usr/bin/env bash
# Submit a MolEditRL-attacking SUCC run: Table1-complete pack, stronger active-property loss,
# many-candidate Table1-success reranking, and strict table metrics.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
export SUCC_DATASET_MODE=moledit
export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v3_attack}"
export SUCC_CONDITION_FEATURES_DIR="${SUCC_CONDITION_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_moledit_hf_vlm_v3_attack}"
export SUCC_HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
export SUCC_MOLEDIT_TRAIN_SPLIT="${SUCC_MOLEDIT_TRAIN_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
export SUCC_MOLEDIT_EVAL_SPLIT="${SUCC_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"

# Keep training focused on MolEditRL Table1 while allowing all rows available for sparse tasks.
export SUCC_MOLEDIT_TABLE1_TASKS_ONLY="${SUCC_MOLEDIT_TABLE1_TASKS_ONLY:-1}"
export SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK="${SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK:-12000}"
export SUCC_MOLEDIT_BALANCED_EVAL_PER_TASK="${SUCC_MOLEDIT_BALANCED_EVAL_PER_TASK:-100}"
export SUCC_MOLEDIT_TRAIN_LIMIT="${SUCC_MOLEDIT_TRAIN_LIMIT:-90000}"
export SUCC_MOLEDIT_EVAL_LIMIT="${SUCC_MOLEDIT_EVAL_LIMIT:-1000}"
export SUCC_MIN_EDIT_SOURCE_TANIMOTO="${SUCC_MIN_EDIT_SOURCE_TANIMOTO:-0.0}"
export SUCC_REQUIRE_EDIT_QUALITY_COLUMNS=0
export SUCC_REQUIRE_EVAL_ORACLE_STRICT=0

# Latent settings: stay on the validated image-VAE residual path, but push edits harder.
export SUCC_LATENT_BACKEND="${SUCC_LATENT_BACKEND:-image_vae}"
export SUCC_LATENT_TARGET_MODE="${SUCC_LATENT_TARGET_MODE:-residual}"
export SUCC_DIFFUSION_OBJECTIVE="${SUCC_DIFFUSION_OBJECTIVE:-pred_x0}"
export SUCC_RESIDUAL_SAMPLE_SCALE="${SUCC_RESIDUAL_SAMPLE_SCALE:-1.45}"
export SUCC_CONNECTOR_LATENT_BLEND="${SUCC_CONNECTOR_LATENT_BLEND:-0.20}"
export SUCC_SAMPLE_STEPS="${SUCC_SAMPLE_STEPS:-48}"
export SUCC_SAMPLE_ETA="${SUCC_SAMPLE_ETA:-0.0}"
export SUCC_IMAGE_VAE_INK_LOSS_WEIGHT="${SUCC_IMAGE_VAE_INK_LOSS_WEIGHT:-4.0}"

# Optimization: Table1 and active properties dominate, because the paper table is the target.
export SUCC_STAGE1_EPOCHS="${SUCC_STAGE1_EPOCHS:-4}"
export SUCC_STAGE2_EPOCHS="${SUCC_STAGE2_EPOCHS:-14}"
export SUCC_STAGE3_EPOCHS="${SUCC_STAGE3_EPOCHS:-6}"
export SUCC_BATCH_SIZE="${SUCC_BATCH_SIZE:-48}"
export SUCC_EVAL_BATCH_SIZE="${SUCC_EVAL_BATCH_SIZE:-64}"
export SUCC_AUX_LOSS_WEIGHT="${SUCC_AUX_LOSS_WEIGHT:-0.90}"
export SUCC_SAMPLING_STRATEGY="${SUCC_SAMPLING_STRATEGY:-weighted}"
export SUCC_TABLE1_SAMPLE_WEIGHT="${SUCC_TABLE1_SAMPLE_WEIGHT:-5.0}"
export SUCC_TRAIN_PROPERTY_SAMPLE_WEIGHTS="${SUCC_TRAIN_PROPERTY_SAMPLE_WEIGHTS:-MW=6,SA=6,RB=4,HBA=4,QED=3,LogP=3}"
export SUCC_AUX_PROPERTY_WEIGHTS="${SUCC_AUX_PROPERTY_WEIGHTS:-MW=5,SA=5,RB=4,HBA=4,QED=3,LogP=3}"
export SUCC_AUX_ALL_PROPERTIES="${SUCC_AUX_ALL_PROPERTIES:-1}"
export SUCC_CONDITION_DROPOUT="${SUCC_CONDITION_DROPOUT:-0.05}"
export SUCC_SOURCE_DROPOUT="${SUCC_SOURCE_DROPOUT:-0.02}"

# OCR stays removed. The benchmark path is direct materialization plus MolEdit table metrics.
export SUCC_RUN_DATASET_EXPORT="${SUCC_RUN_DATASET_EXPORT:-1}"
export SUCC_RUN_FEATURE_EXPORT="${SUCC_RUN_FEATURE_EXPORT:-auto}"
export SUCC_DECODE_EVAL_IMAGES=0
export SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=0
export SUCC_RUN_MOLSCRIBE_OCR=0
export SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER="${SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER:-1}"
export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-table_attack}"
export SUCC_SOURCE_SIMILARITY_RERANK_CANDIDATES="${SUCC_SOURCE_SIMILARITY_RERANK_CANDIDATES:-2048}"
export SUCC_TABLE_SUCCESS_RERANK_CANDIDATES="${SUCC_TABLE_SUCCESS_RERANK_CANDIDATES:-2048}"
export SUCC_TABLE_SUCCESS_WEIGHT="${SUCC_TABLE_SUCCESS_WEIGHT:-100.0}"
export SUCC_TABLE_SOURCE_WEIGHT="${SUCC_TABLE_SOURCE_WEIGHT:-6.0}"
export SUCC_TABLE_LATENT_WEIGHT="${SUCC_TABLE_LATENT_WEIGHT:-1.0}"
export SUCC_MOLEDIT_TABLE_METHOD="${SUCC_MOLEDIT_TABLE_METHOD:-edit_latent_table_success_rerank}"
export SUCC_MOLEDIT_TABLE_OUTPUT_DIR="${SUCC_MOLEDIT_TABLE_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/moledit_table_metrics_attack}"
export SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY="${SUCC_MOLEDIT_TABLE_MISSING_ORACLE_POLICY:-fail}"
export SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1="${SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1:-1}"

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

echo "Submitting SUCC UniVideo MolEdit v3 attack pipeline"
echo "  output_dir=$SUCC_UNIFIED_OUTPUT_DIR"
echo "  table_profile=$SUCC_MATERIALIZED_BENCHMARK_PROFILE"
echo "  table_method=$SUCC_MOLEDIT_TABLE_METHOD"
echo "  balanced_train_per_task=$SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK"
echo "  aux_loss_weight=$SUCC_AUX_LOSS_WEIGHT"
echo "  table_success_rerank_candidates=$SUCC_TABLE_SUCCESS_RERANK_CANDIDATES"
echo "  residual_sample_scale=$SUCC_RESIDUAL_SAMPLE_SCALE"
echo "  connector_latent_blend=$SUCC_CONNECTOR_LATENT_BLEND"

pipeline_output="$(bash "$SCRIPT_DIR/submit_univideo_molecule_pipeline.sh")"
echo "$pipeline_output"
train_job_id="$(echo "$pipeline_output" | sed -n 's/.*univideo_molecule_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"

if [[ -z "$train_job_id" ]]; then
  echo "ERROR: failed to parse v3 training job id." >&2
  exit 1
fi

if [[ "${SUCC_SUBMIT_TABLE1_EXTENSION_AFTER:-1}" == "1" ]]; then
  echo
  echo "Submitting dependent Table1-complete extension"
  SUCC_TABLE1_EVAL_SLURM_DEPENDENCY="afterok:$train_job_id" \
  SUCC_TABLE1_PACK_DIR="${SUCC_TABLE1_PACK_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/dataset/table1_benchmark_synthetic}" \
  SUCC_TABLE1_SYNTHESIZE_MISSING_TASKS="${SUCC_TABLE1_SYNTHESIZE_MISSING_TASKS:-1}" \
  SUCC_TABLE1_SYNTHETIC_MIN_SOURCE_TANIMOTO="${SUCC_TABLE1_SYNTHETIC_MIN_SOURCE_TANIMOTO:-0.4}" \
  SUCC_TABLE1_SYNTHETIC_CANDIDATE_LIMIT="${SUCC_TABLE1_SYNTHETIC_CANDIDATE_LIMIT:-8000}" \
  SUCC_TABLE1_BENCHMARK_OUTPUT_DIR="${SUCC_TABLE1_BENCHMARK_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_table1_attack}" \
  SUCC_TABLE1_TABLE_OUTPUT_DIR="${SUCC_TABLE1_TABLE_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/moledit_table_metrics_table1_attack}" \
  SUCC_MATERIALIZED_BENCHMARK_PROFILE="$SUCC_MATERIALIZED_BENCHMARK_PROFILE" \
  SUCC_MOLEDIT_TABLE_METHOD="$SUCC_MOLEDIT_TABLE_METHOD" \
  SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1=1 \
  SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE="${SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE:-1}" \
  bash "$SCRIPT_DIR/submit_univideo_moledit_table1_extension.sh"
fi

echo
echo "v3 attack pipeline submitted."
echo "  train_job_id=$train_job_id"
echo "  main_table_summary=$SUCC_MOLEDIT_TABLE_OUTPUT_DIR/moledit_table_summary.md"
echo "  table1_attack_summary=$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/moledit_table_metrics_table1_attack/moledit_table_summary.md"
