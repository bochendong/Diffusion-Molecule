#!/usr/bin/env bash
# First-pass unified SMILES generator pipeline:
# prepare rows -> feature export -> SFT+RL+benchmark grid (sample/beam x with_image/no_image).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

export SUCC_PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ACCOUNT="${SUCC_UNIFIED_PIPELINE_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
SUITE_ROOT="${SUCC_UNIFIED_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
DATA_ROOT="${SUCC_UNIFIED_DATA_ROOT:-$SUITE_ROOT/dataset}"
FEATURE_ROOT="${SUCC_UNIFIED_FEATURE_VARIANTS_OUTPUT_ROOT:-$SUITE_ROOT/feature_variants}"

PREP_TIME="${SUCC_UNIFIED_PREP_SLURM_TIME:-02:00:00}"
PREP_MEM="${SUCC_UNIFIED_PREP_SLURM_MEM:-32G}"
FEATURE_TIME="${SUCC_UNIFIED_FEATURE_SLURM_TIME:-12:00:00}"
FEATURE_MEM="${SUCC_UNIFIED_FEATURE_SLURM_MEM:-96G}"
MAIN_TIME="${SUCC_UNIFIED_MAIN_SLURM_TIME:-24:00:00}"
MAIN_MEM="${SUCC_UNIFIED_MAIN_SLURM_MEM:-96G}"
CPUS="${SUCC_UNIFIED_PIPELINE_SLURM_CPUS:-8}"

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

# First-pass conservative settings from gpu_memory_plan.md
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
export SUCC_UNIFIED_SUITE_RUN_FEATURE_EXPORT=0
export SUCC_UNIFIED_SUITE_RUN_TRAIN=1
export SUCC_UNIFIED_SUITE_RUN_RL=1
export SUCC_UNIFIED_SUITE_RUN_BENCHMARK=1

mkdir -p "$LOG_DIR" "$SUITE_ROOT" "$DATA_ROOT" "$FEATURE_ROOT"

echo "Unified SMILES generator first-pass pipeline"
echo "  suite_root=$SUITE_ROOT"
echo "  train_csv=$SUCC_UNIFIED_TRAIN_CSV"
echo "  eval_csv=$SUCC_UNIFIED_EVAL_CSV"
echo "  feature_root=$FEATURE_ROOT"
echo "  epochs=$SUCC_UNIFIED_EPOCHS rl_epochs=$SUCC_UNIFIED_RL_EPOCHS"
echo "  rl_batch=$SUCC_UNIFIED_RL_BATCH_SIZE rollouts=$SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT"
echo "  sample/beam/top_k=$SUCC_UNIFIED_SUITE_SAMPLE_NUM_SAMPLES"
echo "  benchmark_tasks=$SUCC_UNIFIED_BENCHMARK_TASKS"

prep_output="$(
  sbatch \
    --account="$ACCOUNT" \
    --job-name="succ-unified-prep" \
    --time="$PREP_TIME" \
    --mem="$PREP_MEM" \
    --cpus-per-task=4 \
    --output="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    --wrap="bash '$SCRIPT_DIR/run_unified_smiles_generator_prepare_data.sh'" 2>&1
)"
echo "$prep_output"
prep_job="$(echo "$prep_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$prep_job" ]]; then
  echo "ERROR: failed to submit data-prep job." >&2
  exit 1
fi

feature_output="$(
  sbatch \
    --account="$ACCOUNT" \
    --job-name="succ-unified-features" \
    --time="$FEATURE_TIME" \
    --mem="$FEATURE_MEM" \
    --cpus-per-task="$CPUS" \
    --dependency="afterok:$prep_job" \
    --output="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    --wrap="bash '$SCRIPT_DIR/run_unified_smiles_generator_feature_variants.sh'" 2>&1
)"
echo "$feature_output"
feature_job="$(echo "$feature_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$feature_job" ]]; then
  echo "ERROR: failed to submit feature-export job." >&2
  exit 1
fi

main_output="$(
  SUCC_UNIFIED_SUITE_GPU_PROFILE="${SUCC_UNIFIED_SUITE_GPU_PROFILE:-h100_40gb_mig}" \
  SUCC_UNIFIED_SUITE_SLURM_TIME="$MAIN_TIME" \
  SUCC_UNIFIED_SUITE_SLURM_MEM="$MAIN_MEM" \
  SUCC_UNIFIED_SUITE_SLURM_CPUS="$CPUS" \
  SUCC_UNIFIED_SUITE_SLURM_JOB_NAME="succ-unified-firstpass" \
  SUCC_UNIFIED_SUITE_SLURM_DEPENDENCY="afterok:$feature_job" \
  bash "$SCRIPT_DIR/submit_unified_smiles_generator_experiment_suite.sh" 2>&1
)"
echo "$main_output"
main_job="$(echo "$main_output" | sed -n 's/unified_smiles_generator_experiment_suite_job=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "$main_job" ]]; then
  main_job="$(echo "$main_output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
fi
if [[ -z "$main_job" ]]; then
  echo "ERROR: failed to submit main experiment-suite job." >&2
  exit 1
fi

echo
echo "Unified first-pass pipeline submitted."
echo "  prep_job=$prep_job"
echo "  feature_job=$feature_job"
echo "  main_job=$main_job"
echo "  suite_root=$SUITE_ROOT"
