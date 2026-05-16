#!/bin/bash
set -euo pipefail

# Submit the paper-facing verified instruction editing ablations.
#
# Usage:
#   cd PhysTabMol
#   bash scripts/run_instruction_ablation.sh

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"

MODES="${PHYSTABMOL_ABLATION_MODES:-full rule_plan random_plan oracle_plan no_instruction retrieval_only fragment_no_planner}"
ACCOUNT="${PHYSTABMOL_ACCOUNT:-def-hup-ab}"
WALLTIME="${PHYSTABMOL_WALLTIME:-20:00:00}"

echo "Submitting instruction ablations: $MODES"
for mode in $MODES; do
  export PHYSTABMOL_RUN_NAME="${PHYSTABMOL_RUN_PREFIX:-instruction_ablation}_${mode}"
  export PHYSTABMOL_PLANNER_MODE="diffusion"
  export PHYSTABMOL_ENABLE_MMP_DECODER="0"
  export PHYSTABMOL_ENABLE_SOURCE_AWARE_DECODER="0"
  export PHYSTABMOL_DISABLE_INSTRUCTION_FEATURES="0"
  export PHYSTABMOL_DISABLE_INSTRUCTION_GUIDED_PLAN="0"
  case "$mode" in
    full)
      ;;
    rule_plan)
      export PHYSTABMOL_PLANNER_MODE="rule"
      ;;
    random_plan)
      export PHYSTABMOL_PLANNER_MODE="random"
      ;;
    oracle_plan)
      export PHYSTABMOL_PLANNER_MODE="oracle"
      ;;
    no_instruction)
      export PHYSTABMOL_DISABLE_INSTRUCTION_FEATURES="1"
      ;;
    retrieval_only)
      export PHYSTABMOL_ENABLE_MMP_DECODER="1"
      export PHYSTABMOL_ENABLE_SOURCE_AWARE_DECODER="1"
      export PHYSTABMOL_DISABLE_FRAGMENT_GROWTH_DECODER="1"
      ;;
    fragment_no_planner)
      export PHYSTABMOL_PLANNER_MODE="rule"
      ;;
    *)
      echo "Unknown ablation mode: $mode" >&2
      exit 2
      ;;
  esac
  echo "Submitting $mode as $PHYSTABMOL_RUN_NAME"
  sbatch --account="$ACCOUNT" --time="$WALLTIME" scripts/run_instruction_editing_gpu.slurm.sh
  unset PHYSTABMOL_DISABLE_FRAGMENT_GROWTH_DECODER
done
