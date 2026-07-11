#!/usr/bin/env bash
# Evaluate one Unified checkpoint on full 2p-7p, OOD, and Table1 at pinned budgets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

STAGE="${SUCC_UNIFIED_JOINT_STAGE:-u2}"
JOINT_ROOT="${SUCC_UNIFIED_JOINT_ROOT:-SketchMol-Understanding-Condition/outputs/unified_smiles_generator_joint_v2}"
P2P7P_ROOT="${SUCC_UNIFIED_JOINT_2P7P_ROOT:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition}"
TASKS_RAW="${SUCC_UNIFIED_JOINT_EVAL_TASKS:-2p7p,ood,table1}"
BUDGETS_RAW="${SUCC_UNIFIED_JOINT_EVAL_BUDGETS:-20,128}"
SELECTION_MODES_RAW="${SUCC_UNIFIED_JOINT_SELECTION_MODES:-finalizer,raw}"

case "$STAGE" in
u0)
  CHECKPOINT="${SUCC_UNIFIED_CHECKPOINT:-$P2P7P_ROOT/direct_smiles_model/direct_smiles_generator.pt}"
  ;;
u1)
  CHECKPOINT="${SUCC_UNIFIED_CHECKPOINT:-$JOINT_ROOT/u1_joint_sft/unified_smiles_generator.pt}"
  ;;
u2)
  CHECKPOINT="${SUCC_UNIFIED_CHECKPOINT:-$JOINT_ROOT/u2_joint_protected_sft/unified_smiles_generator.pt}"
  ;;
*)
  echo "ERROR: unknown SUCC_UNIFIED_JOINT_STAGE=$STAGE (expected u0, u1, or u2)" >&2
  exit 2
  ;;
esac

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi

IFS=',' read -r -a TASKS <<< "$TASKS_RAW"
IFS=',' read -r -a BUDGETS <<< "$BUDGETS_RAW"
IFS=',' read -r -a SELECTION_MODES <<< "$SELECTION_MODES_RAW"

for task in "${TASKS[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    for selection in "${SELECTION_MODES[@]}"; do
      disable_finalizer=0
      moledit_selection_mode=unified_score
      if [[ "$selection" == "raw" ]]; then
        disable_finalizer=1
        moledit_selection_mode=rank
      elif [[ "$selection" != "finalizer" ]]; then
        echo "ERROR: selection mode must be finalizer or raw, got $selection" >&2
        exit 2
      fi
      output_root="$JOINT_ROOT/eval/$STAGE/$task/at${budget}/$selection"
      echo
      echo "=== Unified Joint v2 eval stage=$STAGE task=$task budget=$budget selection=$selection ==="
      SUCC_UNIFIED_CHECKPOINT="$CHECKPOINT" \
      SUCC_UNIFIED_FAIR_STAGE="$STAGE" \
      SUCC_UNIFIED_FAIR_TASK="$task" \
      SUCC_UNIFIED_FAIR_BUDGET="$budget" \
      SUCC_UNIFIED_FAIR_OUTPUT_ROOT="$output_root" \
      SUCC_UNIFIED_DISABLE_FINALIZER="$disable_finalizer" \
      SUCC_UNIFIED_METHOD_NAME="unified_joint_v2_${STAGE}_${task}_at${budget}_${selection}" \
      SUCC_UNIFIED_MOLEDIT_BUDGETS="$budget" \
      SUCC_UNIFIED_MOLEDIT_SELECTION_MODE="$moledit_selection_mode" \
      bash "$SCRIPT_DIR/run_unified_fair_v1_eval.sh"
    done
  done
done

"${SUCC_PYTHON_BIN:-python3}" "$SCRIPT_DIR/collect_unified_joint_v2_results.py" \
  --eval-root "$JOINT_ROOT/eval" \
  --stage "$STAGE" \
  --output-csv "$JOINT_ROOT/eval/$STAGE/unified_joint_v2_summary.csv"

echo
echo "Unified Joint v2 eval complete: stage=$STAGE checkpoint=$CHECKPOINT"
echo "  combined_summary=$JOINT_ROOT/eval/$STAGE/unified_joint_v2_summary.csv"
