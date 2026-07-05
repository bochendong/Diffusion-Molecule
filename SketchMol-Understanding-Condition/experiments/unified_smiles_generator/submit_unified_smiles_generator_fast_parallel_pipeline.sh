#!/usr/bin/env bash
# Fast parallel unified pipeline:
#   prep (CPU)
#   train-features + eval-features (2 GPU jobs in parallel)
#   with_image + no_image modality suites (2 GPU jobs in parallel)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
export SUCC_HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
ACCOUNT="${SUCC_UNIFIED_PIPELINE_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
SUITE_ROOT="${SUCC_UNIFIED_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
DATA_ROOT="${SUCC_UNIFIED_DATA_ROOT:-$SUITE_ROOT/dataset}"
FEATURE_ROOT="${SUCC_UNIFIED_FEATURE_VARIANTS_OUTPUT_ROOT:-$SUITE_ROOT/feature_variants}"

export SUCC_UNIFIED_SUITE_ROOT="$SUITE_ROOT"
export SUCC_UNIFIED_DATA_ROOT="$DATA_ROOT"
export SUCC_UNIFIED_FEATURE_VARIANTS_OUTPUT_ROOT="$FEATURE_ROOT"
export SUCC_UNIFIED_TRAIN_CSV="${SUCC_UNIFIED_TRAIN_CSV:-$DATA_ROOT/unified_train_rows.csv}"
export SUCC_UNIFIED_EVAL_CSV="${SUCC_UNIFIED_EVAL_CSV:-$DATA_ROOT/unified_eval_rows.csv}"
export SUCC_UNIFIED_TRAIN_FEATURES_DIR="${SUCC_UNIFIED_TRAIN_FEATURES_DIR:-$FEATURE_ROOT/train_condition_features_hf_vlm}"
export SUCC_UNIFIED_EVAL_FEATURES_DIR="${SUCC_UNIFIED_EVAL_FEATURES_DIR:-$FEATURE_ROOT/eval_condition_features_hf_vlm}"
export SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV:-$DATA_ROOT/table1_eval_pack/table1_moledit_rows.csv}"
export SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV="${SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv}"
export SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV="${SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV:-$SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV}"

export SUCC_HF_BATCH_SIZE="${SUCC_HF_BATCH_SIZE:-1}"
export SUCC_UNIFIED_EPOCHS="${SUCC_UNIFIED_EPOCHS:-1}"
export SUCC_UNIFIED_RL_EPOCHS="${SUCC_UNIFIED_RL_EPOCHS:-1}"
export SUCC_UNIFIED_RL_BATCH_SIZE="${SUCC_UNIFIED_RL_BATCH_SIZE:-4}"
export SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT="${SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT:-8}"
export SUCC_UNIFIED_NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-20}"
export SUCC_UNIFIED_BEAM_SIZE="${SUCC_UNIFIED_BEAM_SIZE:-20}"
export SUCC_UNIFIED_TOP_K_CANDIDATES="${SUCC_UNIFIED_TOP_K_CANDIDATES:-20}"
export SUCC_UNIFIED_SUITE_SAMPLE_NUM_SAMPLES="${SUCC_UNIFIED_SUITE_SAMPLE_NUM_SAMPLES:-20}"
export SUCC_UNIFIED_SUITE_BEAM_SIZE="${SUCC_UNIFIED_SUITE_BEAM_SIZE:-20}"
export SUCC_UNIFIED_BENCHMARK_TASKS="${SUCC_UNIFIED_BENCHMARK_TASKS:-denovo_2p7p,denovo_ood,external_multiproperty,moledit_table1}"
export SUCC_UNIFIED_MOLEDIT_BUDGETS="${SUCC_UNIFIED_MOLEDIT_BUDGETS:-20}"
export SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY="${SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY:-skip-task}"
export SUCC_UNIFIED_TABLE1_TRAIN_PER_TASK="${SUCC_UNIFIED_TABLE1_TRAIN_PER_TASK:-200}"

PREP_TIME="${SUCC_UNIFIED_PREP_SLURM_TIME:-02:00:00}"
PREP_MEM="${SUCC_UNIFIED_PREP_SLURM_MEM:-32G}"
FEATURE_TIME="${SUCC_UNIFIED_FEATURE_SLURM_TIME:-12:00:00}"
FEATURE_MEM="${SUCC_UNIFIED_FEATURE_SLURM_MEM:-96G}"
MODALITY_TIME="${SUCC_UNIFIED_MODALITY_SLURM_TIME:-16:00:00}"
MODALITY_MEM="${SUCC_UNIFIED_MODALITY_SLURM_MEM:-64G}"
CPUS="${SUCC_UNIFIED_PIPELINE_SLURM_CPUS:-8}"
FEATURE_GPU="${SUCC_UNIFIED_FEATURE_SLURM_GPUS:-nvidia_h100_80gb_hbm3_3g.40gb:1}"
MODALITY_GPU="${SUCC_UNIFIED_MODALITY_SLURM_GPUS:-nvidia_h100_80gb_hbm3_2g.20gb:1}"

mkdir -p "$LOG_DIR" "$SUITE_ROOT" "$DATA_ROOT" "$FEATURE_ROOT"

submit_cpu() {
  local name="$1"
  local wrap="$2"
  sbatch \
    --account="$ACCOUNT" \
    --job-name="$name" \
    --time="$PREP_TIME" \
    --mem="$PREP_MEM" \
    --cpus-per-task=4 \
    --output="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    --wrap="$wrap"
}

submit_gpu() {
  local name="$1"
  local dependency="$2"
  local wrap="$3"
  local args=(
    --account="$ACCOUNT"
    --job-name="$name"
    --time="$FEATURE_TIME"
    --mem="$FEATURE_MEM"
    --cpus-per-task="$CPUS"
    --gpus="$FEATURE_GPU"
    --output="$LOG_DIR/%x-%j.log"
    --export=ALL
    --wrap="$wrap"
  )
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="$dependency")
  fi
  sbatch "${args[@]}"
}

submit_gpu_modality() {
  local name="$1"
  local dependency="$2"
  local modality="$3"
  sbatch \
    --account="$ACCOUNT" \
    --job-name="$name" \
    --time="$MODALITY_TIME" \
    --mem="$MODALITY_MEM" \
    --cpus-per-task="$CPUS" \
    --gpus="$MODALITY_GPU" \
    --dependency="$dependency" \
    --output="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    --wrap="bash '$SCRIPT_DIR/run_unified_smiles_generator_modality_suite.sh' '$modality'"
}

echo "Unified SMILES fast parallel pipeline"
echo "  suite_root=$SUITE_ROOT"
echo "  feature_gpu=$FEATURE_GPU"
echo "  modality_gpu=$MODALITY_GPU"
echo "  hf_model=$SUCC_HF_MODEL_NAME_OR_PATH"

prep_dep=""
if [[ "${SUCC_UNIFIED_PIPELINE_SKIP_PREP:-0}" == "1" ]]; then
  echo "  skip_prep=1"
else
  prep_out="$(submit_cpu "succ-unified-prep" "bash '$SCRIPT_DIR/run_unified_smiles_generator_prepare_data.sh'")"
  echo "$prep_out"
  prep_job="$(echo "$prep_out" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p')"
  prep_dep="afterok:$prep_job"
fi

feat_train_out="$(submit_gpu "succ-unified-feat-tr" "$prep_dep" "SUCC_UNIFIED_FEATURE_VARIANTS_RUN_TRAIN=1 SUCC_UNIFIED_FEATURE_VARIANTS_RUN_EVAL=0 bash '$SCRIPT_DIR/run_unified_smiles_generator_feature_variants.sh'")"
echo "$feat_train_out"
feat_train_job="$(echo "$feat_train_out" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p')"

feat_eval_out="$(submit_gpu "succ-unified-feat-ev" "$prep_dep" "SUCC_UNIFIED_FEATURE_VARIANTS_RUN_TRAIN=0 SUCC_UNIFIED_FEATURE_VARIANTS_RUN_EVAL=1 bash '$SCRIPT_DIR/run_unified_smiles_generator_feature_variants.sh'")"
echo "$feat_eval_out"
feat_eval_job="$(echo "$feat_eval_out" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p')"

feat_dep="afterok:${feat_train_job}:${feat_eval_job}"

img_out="$(submit_gpu_modality "succ-unified-img" "$feat_dep" "with_image")"
echo "$img_out"
img_job="$(echo "$img_out" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p')"

noimg_out="$(submit_gpu_modality "succ-unified-noimg" "$feat_dep" "no_image")"
echo "$noimg_out"
noimg_job="$(echo "$noimg_out" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p')"

echo
echo "Fast parallel pipeline submitted."
if [[ -n "${prep_job:-}" ]]; then
  echo "  prep_job=$prep_job"
fi
echo "  feat_train_job=$feat_train_job"
echo "  feat_eval_job=$feat_eval_job"
echo "  with_image_job=$img_job"
echo "  no_image_job=$noimg_job"
echo "  suite_root=$SUITE_ROOT"
