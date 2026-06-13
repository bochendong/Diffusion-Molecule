#!/usr/bin/env bash
# Submit dual-mode SUCC training: MolEdit Table1 edits + de novo 2p-7p + OOD rows.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
export SUCC_DATASET_MODE=dualmode
export SUCC_UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_dualmode_v2_guarded}"
export SUCC_CONDITION_FEATURES_DIR="${SUCC_CONDITION_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_moledit_hf_vlm_dualmode_v2_guarded}"
export SUCC_HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
export SUCC_MOLEDIT_TRAIN_SPLIT="${SUCC_MOLEDIT_TRAIN_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
export SUCC_MOLEDIT_EVAL_SPLIT="${SUCC_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
export SUCC_DENOVO_MOLECULE_DB_CSV="${SUCC_DENOVO_MOLECULE_DB_CSV:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1/molecule_database.csv}"

# MolEdit Table1 core
export SUCC_MOLEDIT_TABLE1_TASKS_ONLY="${SUCC_MOLEDIT_TABLE1_TASKS_ONLY:-1}"
export SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK="${SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK:-12000}"
export SUCC_MOLEDIT_BALANCED_EVAL_PER_TASK="${SUCC_MOLEDIT_BALANCED_EVAL_PER_TASK:-100}"
export SUCC_MOLEDIT_TRAIN_LIMIT="${SUCC_MOLEDIT_TRAIN_LIMIT:-90000}"
export SUCC_MOLEDIT_EVAL_LIMIT="${SUCC_MOLEDIT_EVAL_LIMIT:-1000}"

# Dual-mode de novo / OOD mixing
export SUCC_DENOVO_EVAL_ROWS_PER_PROPERTY_COUNT="${SUCC_DENOVO_EVAL_ROWS_PER_PROPERTY_COUNT:-1000}"
export SUCC_DENOVO_TRAIN_ROWS_PER_PROPERTY_COUNT="${SUCC_DENOVO_TRAIN_ROWS_PER_PROPERTY_COUNT:-500}"
export SUCC_OOD_EVAL_ROWS_PER_SPEC="${SUCC_OOD_EVAL_ROWS_PER_SPEC:-100}"
export SUCC_OOD_TRAIN_ROWS_PER_SPEC="${SUCC_OOD_TRAIN_ROWS_PER_SPEC:-400}"

export SUCC_MIN_EDIT_SOURCE_TANIMOTO="${SUCC_MIN_EDIT_SOURCE_TANIMOTO:-0.0}"
export SUCC_REQUIRE_EDIT_QUALITY_COLUMNS=0
export SUCC_REQUIRE_EVAL_ORACLE_STRICT=0

# Latent / optimization: inherit v3 attack defaults, tuned for broader coverage.
export SUCC_LATENT_BACKEND="${SUCC_LATENT_BACKEND:-image_vae}"
export SUCC_LATENT_TARGET_MODE="${SUCC_LATENT_TARGET_MODE:-mixed}"
export SUCC_DIFFUSION_OBJECTIVE="${SUCC_DIFFUSION_OBJECTIVE:-pred_x0}"
export SUCC_RESIDUAL_SAMPLE_SCALE="${SUCC_RESIDUAL_SAMPLE_SCALE:-1.45}"
export SUCC_CONNECTOR_LATENT_BLEND="${SUCC_CONNECTOR_LATENT_BLEND:-0.20}"
export SUCC_SAMPLE_STEPS="${SUCC_SAMPLE_STEPS:-48}"
export SUCC_SAMPLE_ETA="${SUCC_SAMPLE_ETA:-0.0}"
export SUCC_IMAGE_VAE_INK_LOSS_WEIGHT="${SUCC_IMAGE_VAE_INK_LOSS_WEIGHT:-4.0}"

export SUCC_STAGE1_EPOCHS="${SUCC_STAGE1_EPOCHS:-4}"
export SUCC_STAGE2_EPOCHS="${SUCC_STAGE2_EPOCHS:-14}"
export SUCC_STAGE3_EPOCHS="${SUCC_STAGE3_EPOCHS:-6}"
export SUCC_BATCH_SIZE="${SUCC_BATCH_SIZE:-48}"
export SUCC_EVAL_BATCH_SIZE="${SUCC_EVAL_BATCH_SIZE:-64}"
export SUCC_AUX_LOSS_WEIGHT="${SUCC_AUX_LOSS_WEIGHT:-0.90}"
export SUCC_SAMPLING_STRATEGY="${SUCC_SAMPLING_STRATEGY:-weighted}"
export SUCC_TABLE1_SAMPLE_WEIGHT="${SUCC_TABLE1_SAMPLE_WEIGHT:-5.0}"
export SUCC_DENOVO_SAMPLE_WEIGHT="${SUCC_DENOVO_SAMPLE_WEIGHT:-2.0}"
export SUCC_DENOVO_DIVERSITY_LOSS_WEIGHT="${SUCC_DENOVO_DIVERSITY_LOSS_WEIGHT:-0.02}"
export SUCC_DENOVO_DIVERSITY_MARGIN="${SUCC_DENOVO_DIVERSITY_MARGIN:-0.85}"
export SUCC_TRAIN_PROPERTY_SAMPLE_WEIGHTS="${SUCC_TRAIN_PROPERTY_SAMPLE_WEIGHTS:-MW=6,SA=6,RB=4,HBA=4,QED=3,LogP=3}"
export SUCC_AUX_PROPERTY_WEIGHTS="${SUCC_AUX_PROPERTY_WEIGHTS:-MW=5,SA=5,RB=4,HBA=4,QED=3,LogP=3}"
export SUCC_AUX_ALL_PROPERTIES="${SUCC_AUX_ALL_PROPERTIES:-1}"
export SUCC_CONDITION_DROPOUT="${SUCC_CONDITION_DROPOUT:-0.05}"
export SUCC_SOURCE_DROPOUT="${SUCC_SOURCE_DROPOUT:-0.08}"

export SUCC_RUN_DATASET_EXPORT="${SUCC_RUN_DATASET_EXPORT:-1}"
export SUCC_RUN_FEATURE_EXPORT="${SUCC_RUN_FEATURE_EXPORT:-auto}"
export SUCC_DECODE_EVAL_IMAGES=0
export SUCC_RUN_IMAGE_STRUCTURE_BENCHMARK=0
export SUCC_RUN_MOLSCRIBE_OCR=0
export SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER="${SUCC_SUBMIT_MATERIALIZED_BENCHMARK_AFTER:-0}"
export SUCC_SUBMIT_TABLE1_EXTENSION_AFTER="${SUCC_SUBMIT_TABLE1_EXTENSION_AFTER:-1}"
export SUCC_SUBMIT_DENOVO_BENCHMARK_AFTER="${SUCC_SUBMIT_DENOVO_BENCHMARK_AFTER:-1}"
export SUCC_SUBMIT_OOD_BENCHMARK_AFTER="${SUCC_SUBMIT_OOD_BENCHMARK_AFTER:-1}"

# Table1 guard follows the strongest v3 attack evaluation path, so zero-source
# gains are not accepted if the 10-task Acc@0.65 mean regresses.
export SUCC_TABLE1_PER_TASK="${SUCC_TABLE1_PER_TASK:-100}"
export SUCC_TABLE1_SYNTHESIZE_MISSING_TASKS="${SUCC_TABLE1_SYNTHESIZE_MISSING_TASKS:-1}"
export SUCC_TABLE1_SYNTHETIC_MIN_SOURCE_TANIMOTO="${SUCC_TABLE1_SYNTHETIC_MIN_SOURCE_TANIMOTO:-0.4}"
export SUCC_TABLE1_SYNTHETIC_CANDIDATE_LIMIT="${SUCC_TABLE1_SYNTHETIC_CANDIDATE_LIMIT:-8000}"
export SUCC_MATERIALIZED_BENCHMARK_PROFILE="${SUCC_MATERIALIZED_BENCHMARK_PROFILE:-table_attack}"
export SUCC_TABLE_SUCCESS_RERANK_CANDIDATES="${SUCC_TABLE_SUCCESS_RERANK_CANDIDATES:-2048}"
export SUCC_TABLE_SUCCESS_WEIGHT="${SUCC_TABLE_SUCCESS_WEIGHT:-100.0}"
export SUCC_TABLE_SOURCE_WEIGHT="${SUCC_TABLE_SOURCE_WEIGHT:-6.0}"
export SUCC_TABLE_LATENT_WEIGHT="${SUCC_TABLE_LATENT_WEIGHT:-1.0}"
export SUCC_MOLEDIT_TABLE_METHOD="${SUCC_MOLEDIT_TABLE_METHOD:-edit_latent_table_success_rerank}"
export SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1="${SUCC_MOLEDIT_TABLE_INCLUDE_EMPTY_TABLE1:-1}"
export SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE="${SUCC_MOLEDIT_TABLE_REQUIRE_TABLE1_COVERAGE:-1}"
export SUCC_TABLE1_GUARD_AFTER="${SUCC_TABLE1_GUARD_AFTER:-1}"
export SUCC_TABLE1_GUARD_MIN_ACC065="${SUCC_TABLE1_GUARD_MIN_ACC065:-0.894}"
export SUCC_TABLE1_GUARD_REQUIRE_TASKS="${SUCC_TABLE1_GUARD_REQUIRE_TASKS:-10}"

if [[ -z "${SUCC_IMAGE_VAE_CHECKPOINT:-}" ]]; then
  CANONICAL_VAE="SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/molecule_image_vae/molecule_image_vae.pt"
  if [[ -f "$CANONICAL_VAE" ]]; then
    export SUCC_IMAGE_VAE_CHECKPOINT="$CANONICAL_VAE"
    export SUCC_IMAGE_VAE_DIR="$(dirname "$CANONICAL_VAE")"
    export SUCC_RUN_IMAGE_VAE_TRAIN="${SUCC_RUN_IMAGE_VAE_TRAIN:-0}"
  fi
fi
export SUCC_RUN_IMAGE_VAE_TRAIN="${SUCC_RUN_IMAGE_VAE_TRAIN:-auto}"

echo "Submitting SUCC UniVideo MolEdit dual-mode pipeline"
echo "  output_dir=$SUCC_UNIFIED_OUTPUT_DIR"
echo "  dataset_mode=$SUCC_DATASET_MODE"
echo "  denovo_train_rows_per_property_count=$SUCC_DENOVO_TRAIN_ROWS_PER_PROPERTY_COUNT"
echo "  ood_train_rows_per_spec=$SUCC_OOD_TRAIN_ROWS_PER_SPEC"
echo "  latent_target_mode=$SUCC_LATENT_TARGET_MODE"
echo "  source_dropout=$SUCC_SOURCE_DROPOUT"
echo "  denovo_sample_weight=$SUCC_DENOVO_SAMPLE_WEIGHT"
echo "  denovo_diversity_loss_weight=$SUCC_DENOVO_DIVERSITY_LOSS_WEIGHT"
echo "  balanced_train_per_task=$SUCC_MOLEDIT_BALANCED_TRAIN_PER_TASK"
echo "  submit_table1_extension_after=$SUCC_SUBMIT_TABLE1_EXTENSION_AFTER"
echo "  submit_denovo_benchmark_after=$SUCC_SUBMIT_DENOVO_BENCHMARK_AFTER"
echo "  submit_ood_benchmark_after=$SUCC_SUBMIT_OOD_BENCHMARK_AFTER"
echo "  table1_guard_min_acc065=$SUCC_TABLE1_GUARD_MIN_ACC065"

pipeline_output="$(bash "$SCRIPT_DIR/submit_univideo_molecule_pipeline.sh")"
echo "$pipeline_output"
train_job_id="$(echo "$pipeline_output" | sed -n 's/.*univideo_molecule_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$train_job_id" ]]; then
  train_job_id="$(echo "$pipeline_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
fi
if [[ -z "$train_job_id" ]]; then
  echo "ERROR: failed to parse dual-mode training job id." >&2
  exit 1
fi

train_dependency="afterok:$train_job_id"
echo
echo "Dual-mode training submitted."
echo "  train_job_id=$train_job_id"
echo "  downstream_dependency=$train_dependency"

if [[ "$SUCC_SUBMIT_TABLE1_EXTENSION_AFTER" == "1" ]]; then
  export SUCC_TABLE1_EVAL_SLURM_DEPENDENCY="$train_dependency"
  export SUCC_TABLE1_PACK_DIR="${SUCC_TABLE1_PACK_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/dataset/table1_benchmark_guarded}"
  export SUCC_TABLE1_EVAL_OUTPUT_DIR="${SUCC_TABLE1_EVAL_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/table1_eval_latent_guarded}"
  export SUCC_TABLE1_BENCHMARK_OUTPUT_DIR="${SUCC_TABLE1_BENCHMARK_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/benchmark_materialized_table1_guarded}"
  export SUCC_TABLE1_TABLE_OUTPUT_DIR="${SUCC_TABLE1_TABLE_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR/univideo_molecule/moledit_table_metrics_table1_guarded}"
  echo
  echo "Submitting dependent Table1 guarded extension"
  bash "$SCRIPT_DIR/submit_univideo_moledit_table1_extension.sh"
fi

if [[ "$SUCC_SUBMIT_OOD_BENCHMARK_AFTER" == "1" ]]; then
  export SUCC_OOD_SLURM_DEPENDENCY="$train_dependency"
  export SUCC_OOD_MODEL_OUTPUT_DIR="${SUCC_OOD_MODEL_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR}"
  export SUCC_OOD_OUTPUT_DIR="${SUCC_OOD_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/denovo_ood_ours_dualmode_v2_guarded}"
  echo
  echo "Submitting dependent OOD benchmark"
  bash "$SCRIPT_DIR/submit_denovo_ood_ours_benchmark.sh"
fi

if [[ "$SUCC_SUBMIT_DENOVO_BENCHMARK_AFTER" == "1" ]]; then
  export SUCC_DENOVO_SLURM_DEPENDENCY="$train_dependency"
  export SUCC_DENOVO_MODEL_OUTPUT_DIR="${SUCC_DENOVO_MODEL_OUTPUT_DIR:-$SUCC_UNIFIED_OUTPUT_DIR}"
  export SUCC_DENOVO_OUTPUT_DIR="${SUCC_DENOVO_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/denovo_2p7p_ours_dualmode_v2_guarded}"
  echo
  echo "Submitting dependent de novo 2p-7p benchmark"
  bash "$SCRIPT_DIR/submit_denovo_2p7p_ours_benchmark.sh"
fi
