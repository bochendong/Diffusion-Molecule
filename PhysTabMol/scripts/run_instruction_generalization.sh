#!/bin/bash
set -euo pipefail

# Submit generalization tests for verified instruction editing.
#
# Usage:
#   cd PhysTabMol
#   bash scripts/run_instruction_generalization.sh
#
# Optional:
#   PHYSTABMOL_GENERALIZATION_MODES="template_paraphrase edit_combo scaffold" bash scripts/run_instruction_generalization.sh
#   PHYSTABMOL_LLM_VERIFIED_OUT=data/instruction_editing_llm_verified.csv \
#     PHYSTABMOL_GENERALIZATION_MODES="llm_verified" bash scripts/run_instruction_generalization.sh

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"

MODES="${PHYSTABMOL_GENERALIZATION_MODES:-template_paraphrase edit_combo scaffold llm_verified}"
ACCOUNT="${PHYSTABMOL_ACCOUNT:-def-hup-ab}"
WALLTIME="${PHYSTABMOL_WALLTIME:-20:00:00}"
BASE_DATASET="${PHYSTABMOL_INSTRUCTION_DATASET:-data/instruction_editing.csv}"
LLM_DATASET="${PHYSTABMOL_LLM_VERIFIED_OUT:-data/instruction_editing_llm_verified.csv}"

echo "Submitting instruction generalization modes: $MODES"
for mode in $MODES; do
  export PHYSTABMOL_RUN_NAME="${PHYSTABMOL_RUN_PREFIX:-instruction_generalization}_${mode}"
  export PHYSTABMOL_INSTRUCTION_DATASET="$BASE_DATASET"
  export PHYSTABMOL_SPLIT_COLUMN="split"
  export PHYSTABMOL_PLANNER_MODE="diffusion"
  export PHYSTABMOL_ENABLE_MMP_DECODER="0"
  export PHYSTABMOL_ENABLE_SOURCE_AWARE_DECODER="0"
  export PHYSTABMOL_DISABLE_FRAGMENT_GROWTH_DECODER="0"
  export PHYSTABMOL_DISABLE_INSTRUCTION_FEATURES="0"
  export PHYSTABMOL_DISABLE_INSTRUCTION_GUIDED_PLAN="0"
  export PHYSTABMOL_BLIND_INSTRUCTION="0"
  export PHYSTABMOL_MULTIMODAL_CONTEXT="${PHYSTABMOL_BASE_MULTIMODAL_CONTEXT:-source_reference}"
  case "$mode" in
    template_paraphrase)
      export PHYSTABMOL_SPLIT_COLUMN="paraphrase_split"
      ;;
    edit_combo)
      export PHYSTABMOL_SPLIT_COLUMN="edit_combo_split"
      ;;
    scaffold)
      export PHYSTABMOL_SPLIT_COLUMN="split_by_scaffold"
      ;;
    llm_verified)
      if [[ ! -s "$LLM_DATASET" ]]; then
        echo "Skipping llm_verified: $LLM_DATASET not found. Run scripts/filter_instruction_paraphrases.sh first." >&2
        continue
      fi
      export PHYSTABMOL_INSTRUCTION_DATASET="$LLM_DATASET"
      export PHYSTABMOL_SPLIT_COLUMN="${PHYSTABMOL_LLM_SPLIT_COLUMN:-split}"
      ;;
    *)
      echo "Unknown generalization mode: $mode" >&2
      exit 2
      ;;
  esac
  echo "Submitting $mode as $PHYSTABMOL_RUN_NAME (dataset=$PHYSTABMOL_INSTRUCTION_DATASET split=$PHYSTABMOL_SPLIT_COLUMN)"
  sbatch --account="$ACCOUNT" --time="$WALLTIME" scripts/run_instruction_editing_gpu.slurm.sh
done
