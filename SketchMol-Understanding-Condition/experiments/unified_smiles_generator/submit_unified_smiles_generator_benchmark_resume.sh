#!/usr/bin/env bash
# Resume failed modality jobs after SFT/RL + partial sample benchmarks.
# Finishes sample/moledit eval, then runs beam decode + benchmark.

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
ACCOUNT="${SUCC_UNIFIED_PIPELINE_SLURM_ACCOUNT:-${SUCC_SLURM_ACCOUNT:-def-hup-ab_gpu}}"
LOG_DIR="${SUCC_LOG_DIR:-$PROJECT_DIR/logs}"
SUITE_ROOT="${SUCC_UNIFIED_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
MODALITY_GPU="${SUCC_UNIFIED_MODALITY_SLURM_GPUS:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
MODALITY_TIME="${SUCC_UNIFIED_MODALITY_SLURM_TIME:-08:00:00}"
MODALITY_MEM="${SUCC_UNIFIED_MODALITY_SLURM_MEM:-64G}"
CPUS="${SUCC_UNIFIED_PIPELINE_SLURM_CPUS:-8}"

export SUCC_UNIFIED_SUITE_ROOT="$SUITE_ROOT"
export SUCC_UNIFIED_TRAIN_CSV="${SUCC_UNIFIED_TRAIN_CSV:-$SUITE_ROOT/dataset/unified_train_rows.csv}"
export SUCC_UNIFIED_EVAL_CSV="${SUCC_UNIFIED_EVAL_CSV:-$SUITE_ROOT/dataset/unified_eval_rows.csv}"
export SUCC_UNIFIED_TRAIN_FEATURES_DIR="${SUCC_UNIFIED_TRAIN_FEATURES_DIR:-$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm}"
export SUCC_UNIFIED_EVAL_FEATURES_DIR="${SUCC_UNIFIED_EVAL_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
export SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV:-$SUITE_ROOT/dataset/table1_eval_pack/table1_moledit_rows.csv}"
export SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV="${SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv}"
export SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV="${SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV:-$SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV}"
export SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY="${SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY:-skip-task}"
export SUCC_UNIFIED_BENCHMARK_TASKS="${SUCC_UNIFIED_BENCHMARK_TASKS:-denovo_2p7p,denovo_ood,external_multiproperty,moledit_table1}"
export SUCC_UNIFIED_MOLEDIT_BUDGETS="${SUCC_UNIFIED_MOLEDIT_BUDGETS:-20}"
export SUCC_UNIFIED_SUITE_SAMPLE_NUM_SAMPLES="${SUCC_UNIFIED_SUITE_SAMPLE_NUM_SAMPLES:-20}"
export SUCC_UNIFIED_SUITE_BEAM_SIZE="${SUCC_UNIFIED_SUITE_BEAM_SIZE:-20}"
export SUCC_UNIFIED_NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-20}"
export SUCC_UNIFIED_BEAM_SIZE="${SUCC_UNIFIED_BEAM_SIZE:-20}"
export SUCC_UNIFIED_TOP_K_CANDIDATES="${SUCC_UNIFIED_TOP_K_CANDIDATES:-20}"

submit_resume() {
  local modality="$1"
  local variant=""
  case "$modality" in
    with_image) variant="full" ;;
    no_image) variant="text_only" ;;
    *)
      echo "ERROR: unsupported modality=$modality" >&2
      exit 2
      ;;
  esac

  local checkpoint="$SUITE_ROOT/$modality/group_rl/unified_smiles_generator_group_rl.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: missing checkpoint for $modality: $checkpoint" >&2
    exit 2
  fi

  local sample_dir="$SUITE_ROOT/$modality/benchmark_sample"
  local sample_outputs="$sample_dir/sample_outputs"
  local wrap
  wrap="$(cat <<EOF
set -euo pipefail
cd '$REPO_DIR'

echo '=== resume sample moledit eval ($modality) ==='
SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=0 \
SUCC_UNIFIED_BENCHMARK_TASKS=moledit_table1 \
SUCC_UNIFIED_CHECKPOINT='$checkpoint' \
SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR='$sample_dir' \
SUCC_UNIFIED_SAMPLE_OUTPUT_DIR='$sample_outputs' \
SUCC_UNIFIED_BENCHMARK_PREDICTION_CSV='$sample_outputs/unified_smiles_predictions.csv' \
SUCC_UNIFIED_BENCHMARK_CANDIDATE_CSV='$sample_outputs/unified_smiles_candidate_predictions.csv' \
SUCC_UNIFIED_CONDITION_FEATURE_VARIANT='$variant' \
SUCC_UNIFIED_INPUT_MODALITY='$modality' \
SUCC_UNIFIED_METHOD_NAME='unified_smiles_generator_${modality}_sample' \
SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY='${SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY}' \
SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV='${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV}' \
SUCC_UNIFIED_MOLEDIT_BUDGETS='${SUCC_UNIFIED_MOLEDIT_BUDGETS}' \
SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV='${SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV}' \
SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV='${SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV}' \
bash '$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh'

echo '=== resume beam decode + benchmark ($modality) ==='
SUCC_UNIFIED_SUITE_MODALITIES='$modality' \
SUCC_UNIFIED_SUITE_DECODING_MODES=beam \
SUCC_UNIFIED_SUITE_RUN_FEATURE_EXPORT=0 \
SUCC_UNIFIED_SUITE_RUN_TRAIN=0 \
SUCC_UNIFIED_SUITE_RUN_RL=0 \
SUCC_UNIFIED_SUITE_RUN_BENCHMARK=1 \
SUCC_UNIFIED_${modality^^}_CHECKPOINT='$checkpoint' \
bash '$SCRIPT_DIR/run_unified_smiles_generator_experiment_suite.sh'
EOF
)"

  sbatch \
    --account="$ACCOUNT" \
    --job-name="succ-unified-resume-${modality}" \
    --time="$MODALITY_TIME" \
    --mem="$MODALITY_MEM" \
    --cpus-per-task="$CPUS" \
    --gpus="$MODALITY_GPU" \
    --output="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    --wrap="$wrap"
}

echo "Unified SMILES benchmark resume"
echo "  suite_root=$SUITE_ROOT"
echo "  modality_gpu=$MODALITY_GPU"
echo "  moledit_missing_oracle_policy=$SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY"

for modality in with_image no_image; do
  out="$(submit_resume "$modality")"
  echo "$out"
done
