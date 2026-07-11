#!/usr/bin/env bash
# Unified Fair Protocol v1 P0 suite: full eval sets at fair candidate budgets.
#
# Default (SUCC_UNIFIED_FAIR_SUITE=p0):
#   2p7p  6000 @40  direct_compat
#   ood   1000 @40  direct_compat
#   table1 1000 @20 direct_edit_compat
#
# Set SUCC_UNIFIED_FAIR_STAGE=u1 after U1 training completes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SUITE="${SUCC_UNIFIED_FAIR_SUITE:-p0}"
STAGE="${SUCC_UNIFIED_FAIR_STAGE:-u0}"
INPUT_MODALITY="${SUCC_UNIFIED_INPUT_MODALITY:-with_image}"

run_task() {
  local task="$1"
  echo
  echo "========================================"
  echo "Unified Fair v1: stage=$STAGE task=$task modality=$INPUT_MODALITY"
  echo "========================================"
  SUCC_UNIFIED_FAIR_TASK="$task" \
  SUCC_UNIFIED_FAIR_STAGE="$STAGE" \
  SUCC_UNIFIED_INPUT_MODALITY="$INPUT_MODALITY" \
  bash "$SCRIPT_DIR/run_unified_fair_v1_eval.sh"
}

case "$SUITE" in
p0)
  run_task 2p7p
  run_task ood
  run_task table1
  ;;
2p7p|ood|table1)
  run_task "$SUITE"
  ;;
*)
  echo "ERROR: unknown SUCC_UNIFIED_FAIR_SUITE=$SUITE (expected p0, 2p7p, ood, or table1)" >&2
  exit 2
  ;;
esac

echo
echo "Unified Fair v1 suite complete: stage=$STAGE suite=$SUITE modality=$INPUT_MODALITY"
