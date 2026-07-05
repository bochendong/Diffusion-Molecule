#!/usr/bin/env bash
# GRPO RL + benchmark for both modalities, reusing existing SFT checkpoints.
# Writes to group_rl_grpo/ and benchmark_{sample,beam}_grpo/ to avoid overwriting group_pg runs.

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
JOB_TIME="${SUCC_UNIFIED_GRPO_SLURM_TIME:-12:00:00}"
JOB_MEM="${SUCC_UNIFIED_GRPO_SLURM_MEM:-64G}"
CPUS="${SUCC_UNIFIED_PIPELINE_SLURM_CPUS:-8}"

export SUCC_UNIFIED_SUITE_ROOT="$SUITE_ROOT"
export SUCC_UNIFIED_TRAIN_CSV="${SUCC_UNIFIED_TRAIN_CSV:-$SUITE_ROOT/dataset/unified_train_rows.csv}"
export SUCC_UNIFIED_EVAL_CSV="${SUCC_UNIFIED_EVAL_CSV:-$SUITE_ROOT/dataset/unified_eval_rows.csv}"
export SUCC_UNIFIED_TRAIN_FEATURES_DIR="${SUCC_UNIFIED_TRAIN_FEATURES_DIR:-$SUITE_ROOT/feature_variants/train_condition_features_hf_vlm}"
export SUCC_UNIFIED_EVAL_FEATURES_DIR="${SUCC_UNIFIED_EVAL_FEATURES_DIR:-$SUITE_ROOT/feature_variants/eval_condition_features_hf_vlm}"
export SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV:-$SUITE_ROOT/dataset/table1_eval_pack/table1_moledit_rows.csv}"
export SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV="${SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV:-SketchMol-Understanding-Condition/outputs/external_oracle_build_v1/generated_properties.csv}"
export SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV="${SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV:-$SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV}"

export SUCC_UNIFIED_RL_OBJECTIVE="${SUCC_UNIFIED_RL_OBJECTIVE:-grpo}"
export SUCC_UNIFIED_RL_GRPO_CLIP_EPS="${SUCC_UNIFIED_RL_GRPO_CLIP_EPS:-0.2}"
export SUCC_UNIFIED_RL_GRPO_UPDATE_EPOCHS="${SUCC_UNIFIED_RL_GRPO_UPDATE_EPOCHS:-2}"
export SUCC_UNIFIED_RL_EPOCHS="${SUCC_UNIFIED_RL_EPOCHS:-1}"
export SUCC_UNIFIED_RL_BATCH_SIZE="${SUCC_UNIFIED_RL_BATCH_SIZE:-4}"
export SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT="${SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT:-8}"
export SUCC_UNIFIED_NUM_SAMPLES="${SUCC_UNIFIED_NUM_SAMPLES:-20}"
export SUCC_UNIFIED_BEAM_SIZE="${SUCC_UNIFIED_BEAM_SIZE:-20}"
export SUCC_UNIFIED_TOP_K_CANDIDATES="${SUCC_UNIFIED_TOP_K_CANDIDATES:-20}"
export SUCC_UNIFIED_BENCHMARK_TASKS="${SUCC_UNIFIED_BENCHMARK_TASKS:-denovo_2p7p,denovo_ood,external_multiproperty,moledit_table1}"
export SUCC_UNIFIED_MOLEDIT_BUDGETS="${SUCC_UNIFIED_MOLEDIT_BUDGETS:-20}"
export SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY="${SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY:-skip-task}"

submit_grpo_modality() {
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

  local sft_ckpt="$SUITE_ROOT/$modality/sft/unified_smiles_generator.pt"
  if [[ ! -f "$sft_ckpt" ]]; then
    echo "ERROR: missing SFT checkpoint for $modality: $sft_ckpt" >&2
    exit 2
  fi

  local grpo_dir="$SUITE_ROOT/$modality/group_rl_grpo"
  local wrap
  wrap="$(cat <<EOF
set -euo pipefail
cd '$REPO_DIR'

echo '=== GRPO RL ($modality) ==='
SUCC_UNIFIED_RL_OBJECTIVE='${SUCC_UNIFIED_RL_OBJECTIVE}' \
SUCC_UNIFIED_RL_GRPO_CLIP_EPS='${SUCC_UNIFIED_RL_GRPO_CLIP_EPS}' \
SUCC_UNIFIED_RL_GRPO_UPDATE_EPOCHS='${SUCC_UNIFIED_RL_GRPO_UPDATE_EPOCHS}' \
SUCC_UNIFIED_RL_EPOCHS='${SUCC_UNIFIED_RL_EPOCHS}' \
SUCC_UNIFIED_RL_BATCH_SIZE='${SUCC_UNIFIED_RL_BATCH_SIZE}' \
SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT='${SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT}' \
SUCC_UNIFIED_RL_TRAIN_CSV='${SUCC_UNIFIED_TRAIN_CSV}' \
SUCC_UNIFIED_RL_EVAL_CSV='${SUCC_UNIFIED_EVAL_CSV}' \
SUCC_UNIFIED_RL_OUTPUT_DIR='$grpo_dir' \
SUCC_UNIFIED_RL_RESUME_CHECKPOINT='$sft_ckpt' \
SUCC_UNIFIED_RL_TRAIN_FEATURES_DIR='${SUCC_UNIFIED_TRAIN_FEATURES_DIR}' \
SUCC_UNIFIED_RL_EVAL_FEATURES_DIR='${SUCC_UNIFIED_EVAL_FEATURES_DIR}' \
SUCC_UNIFIED_CONDITION_FEATURE_VARIANT='$variant' \
SUCC_UNIFIED_INPUT_MODALITY='$modality' \
bash '$SCRIPT_DIR/run_unified_smiles_generator_group_rl.sh'

GRPO_CKPT='$grpo_dir/unified_smiles_generator_group_rl.pt'
if [[ ! -f "\$GRPO_CKPT" ]]; then
  echo "ERROR: missing GRPO checkpoint: \$GRPO_CKPT" >&2
  exit 2
fi

for decoding in sample beam; do
  echo "=== GRPO benchmark ($modality, \$decoding) ==="
  OUT_DIR='$SUITE_ROOT/$modality/benchmark_'\${decoding}'_grpo'
  SAMPLE_DIR="\$OUT_DIR/sample_outputs"
  if [[ "\$decoding" == "sample" ]]; then
    NUM_SAMPLES='${SUCC_UNIFIED_NUM_SAMPLES}'
    BEAM_FOR_RUN=1
  else
    NUM_SAMPLES=1
    BEAM_FOR_RUN='${SUCC_UNIFIED_BEAM_SIZE}'
  fi
  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=1 \
  SUCC_UNIFIED_CHECKPOINT="\$GRPO_CKPT" \
  SUCC_UNIFIED_EVAL_CSV='${SUCC_UNIFIED_EVAL_CSV}' \
  SUCC_UNIFIED_EVAL_FEATURES_DIR='${SUCC_UNIFIED_EVAL_FEATURES_DIR}' \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="\$OUT_DIR" \
  SUCC_UNIFIED_SAMPLE_OUTPUT_DIR="\$SAMPLE_DIR" \
  SUCC_UNIFIED_BENCHMARK_TASKS='${SUCC_UNIFIED_BENCHMARK_TASKS}' \
  SUCC_UNIFIED_CONDITION_FEATURE_VARIANT='$variant' \
  SUCC_UNIFIED_INPUT_MODALITY='$modality' \
  SUCC_UNIFIED_METHOD_NAME="unified_smiles_generator_${modality}_\${decoding}_grpo" \
  SUCC_UNIFIED_DECODING_MODE="\$decoding" \
  SUCC_UNIFIED_NUM_SAMPLES="\$NUM_SAMPLES" \
  SUCC_UNIFIED_BEAM_SIZE="\$BEAM_FOR_RUN" \
  SUCC_UNIFIED_TOP_K_CANDIDATES='${SUCC_UNIFIED_TOP_K_CANDIDATES}' \
  SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV='${SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV}' \
  SUCC_UNIFIED_MOLEDIT_BUDGETS='${SUCC_UNIFIED_MOLEDIT_BUDGETS}' \
  SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY='${SUCC_UNIFIED_MOLEDIT_MISSING_ORACLE_POLICY}' \
  SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV='${SUCC_UNIFIED_EXTERNAL_GENERATED_PROPERTIES_CSV}' \
  SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV='${SUCC_UNIFIED_EXTERNAL_SOURCE_PROPERTIES_CSV}' \
  bash '$SCRIPT_DIR/run_unified_smiles_generator_benchmark_suite.sh'
done

echo 'GRPO pipeline complete for $modality'
echo "  grpo_checkpoint=\$GRPO_CKPT"
EOF
)"

  sbatch \
    --account="$ACCOUNT" \
    --job-name="succ-unified-grpo-${modality}" \
    --time="$JOB_TIME" \
    --mem="$JOB_MEM" \
    --cpus-per-task="$CPUS" \
    --gpus="$MODALITY_GPU" \
    --output="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    --wrap="$wrap"
}

echo "Unified SMILES GRPO pipeline"
echo "  suite_root=$SUITE_ROOT"
echo "  modality_gpu=$MODALITY_GPU"
echo "  rl_objective=$SUCC_UNIFIED_RL_OBJECTIVE"
echo "  grpo_clip_eps=$SUCC_UNIFIED_RL_GRPO_CLIP_EPS"
echo "  grpo_update_epochs=$SUCC_UNIFIED_RL_GRPO_UPDATE_EPOCHS"
echo "  rl_batch=$SUCC_UNIFIED_RL_BATCH_SIZE rollouts=$SUCC_UNIFIED_RL_ROLLOUTS_PER_PROMPT"

for modality in with_image no_image; do
  out="$(submit_grpo_modality "$modality")"
  echo "$out"
done
