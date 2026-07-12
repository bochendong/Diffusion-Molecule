#!/usr/bin/env bash
# Evaluate U0 or every U1/U2 epoch on validation@20 using one fixed sampling seed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

STAGE="${SUCC_UNIFIED_JOINT_STAGE:-u2}"
TRAIN_SEED="${SUCC_UNIFIED_TRAIN_SEED:-${SUCC_UNIFIED_SEED:-7}}"
VALIDATION_SEEDS_RAW="${SUCC_UNIFIED_VALIDATION_SEEDS:-${SUCC_UNIFIED_VALIDATION_SEED:-101,202,303}}"
JOINT_ROOT="${SUCC_UNIFIED_JOINT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2}"
SUITE_ROOT="${SUCC_UNIFIED_JOINT_SOURCE_SUITE_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_suite_v1}"
P2P7P_ROOT="${SUCC_UNIFIED_JOINT_2P7P_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
DATA_ROOT="${SUCC_UNIFIED_JOINT_DATA_ROOT:-$JOINT_ROOT/dataset}"
VALIDATION_FEATURES_DIR="${SUCC_UNIFIED_JOINT_VALIDATION_FEATURES_DIR:-$JOINT_ROOT/feature_variants/validation_condition_features_hf_vlm}"
DENOVO_VALIDATION_CSV="${SUCC_UNIFIED_JOINT_DENOVO_VALIDATION_CSV:-$DATA_ROOT/denovo_validation_rows.csv}"
TABLE1_VALIDATION_CSV="${SUCC_UNIFIED_JOINT_TABLE1_VALIDATION_CSV:-$DATA_ROOT/table1_validation_rows.csv}"
BASE_CHECKPOINT="${SUCC_UNIFIED_JOINT_BASE_CHECKPOINT:-$P2P7P_ROOT/direct_smiles_model/direct_smiles_generator.pt}"

case "$STAGE" in
u0)
  CHECKPOINTS=("$BASE_CHECKPOINT")
  OUTPUT_NAMES=(base)
  VALIDATION_ROOT="$JOINT_ROOT/validation/u0"
  ;;
u1)
  CHECKPOINT_DIR="$JOINT_ROOT/u1_joint_sft/seed_${TRAIN_SEED}"
  CHECKPOINTS=()
  while IFS= read -r checkpoint; do CHECKPOINTS+=("$checkpoint"); done < <(
    find "$CHECKPOINT_DIR" -maxdepth 1 -name 'checkpoint_epoch_*.pt' -print | sort
  )
  OUTPUT_NAMES=()
  for checkpoint in "${CHECKPOINTS[@]}"; do OUTPUT_NAMES+=("$(basename "$checkpoint" .pt)"); done
  VALIDATION_ROOT="$JOINT_ROOT/validation/u1/seed_${TRAIN_SEED}"
  ;;
u2)
  CHECKPOINT_DIR="$JOINT_ROOT/u2_joint_protected_sft/seed_${TRAIN_SEED}"
  CHECKPOINTS=()
  while IFS= read -r checkpoint; do CHECKPOINTS+=("$checkpoint"); done < <(
    find "$CHECKPOINT_DIR" -maxdepth 1 -name 'checkpoint_epoch_*.pt' -print | sort
  )
  OUTPUT_NAMES=()
  for checkpoint in "${CHECKPOINTS[@]}"; do OUTPUT_NAMES+=("$(basename "$checkpoint" .pt)"); done
  VALIDATION_ROOT="$JOINT_ROOT/validation/u2/seed_${TRAIN_SEED}"
  ;;
*)
  echo "ERROR: stage must be u0, u1, or u2" >&2
  exit 2
  ;;
esac

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "ERROR: no checkpoints found for stage=$STAGE train_seed=$TRAIN_SEED" >&2
  exit 2
fi

IFS=',' read -r -a VALIDATION_SEEDS <<< "$VALIDATION_SEEDS_RAW"
for index in "${!CHECKPOINTS[@]}"; do
  checkpoint="${CHECKPOINTS[$index]}"
  name="${OUTPUT_NAMES[$index]}"
  for eval_seed in "${VALIDATION_SEEDS[@]}"; do
    for task in 2p7p table1; do
      if [[ "$task" == 2p7p ]]; then
        eval_csv="$DENOVO_VALIDATION_CSV"
        reference_csv=""
      else
        eval_csv="$TABLE1_VALIDATION_CSV"
        reference_csv="$TABLE1_VALIDATION_CSV"
      fi
      output_root="$VALIDATION_ROOT/$name/eval_seed_${eval_seed}/$task"
      SUCC_UNIFIED_JOINT_EVAL_TASK="$task" \
      SUCC_UNIFIED_CHECKPOINT="$checkpoint" \
      SUCC_UNIFIED_EVAL_CSV="$eval_csv" \
      SUCC_UNIFIED_EVAL_FEATURES_DIR="$VALIDATION_FEATURES_DIR" \
      SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$output_root" \
      SUCC_UNIFIED_CANDIDATE_BUDGETS=20 \
      SUCC_UNIFIED_MAX_CANDIDATES=20 \
      SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
      SUCC_UNIFIED_SEED="$eval_seed" \
      SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$reference_csv" \
      SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
      bash "$SCRIPT_DIR/run_unified_joint_v2_eval_one.sh"
    done
  done
done

echo "Unified Joint v2 validation complete: stage=$STAGE train_seed=$TRAIN_SEED eval_seeds=$VALIDATION_SEEDS_RAW"
