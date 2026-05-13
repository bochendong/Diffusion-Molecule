#!/bin/bash
set -euo pipefail

# Submit structure-prompt decoder ablations for an existing trained run.
#
# Usage from PhysTabMol repo root:
#   PHYSTABMOL_RUN_DIR=runs/20260512_235957_sketchmol_comparable_structure_v1 \
#     bash scripts/run_structure_prompt_ablation.sh
#
# Override the mode list, for example:
#   PHYSTABMOL_STRUCTURE_ABLATION_MODES="no_two_step no_condition_guided" bash scripts/run_structure_prompt_ablation.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${PHYSTABMOL_RUN_DIR:-$(find runs -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)}"
MODES="${PHYSTABMOL_STRUCTURE_ABLATION_MODES:-full no_two_step no_condition_guided pair_only}"
CONDITIONS="${PHYSTABMOL_STRUCTURE_PROMPT_CONDITIONS:-0}"
SAMPLES="${PHYSTABMOL_STRUCTURE_PROMPT_SAMPLES:-0}"
DECODE_TOP_K="${PHYSTABMOL_STRUCTURE_PROMPT_DECODE_TOP_K:-0}"
WALLTIME="${PHYSTABMOL_WALLTIME:-08:00:00}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR"
  exit 1
fi

echo "Submitting structure-prompt ablations for run_dir=$RUN_DIR"
echo "modes=$MODES"

for mode in $MODES; do
  case "$mode" in
    full)
      EXTRA_ARGS=()
      ;;
    no_two_step)
      EXTRA_ARGS=(--mmp-fragment-growth-steps 1)
      ;;
    no_condition_guided)
      EXTRA_ARGS=(--disable-structure-prompt-condition-guided-ranking)
      ;;
    pair_only)
      EXTRA_ARGS=(--mmp-fragment-neighbors 0 --mmp-fragment-growth-steps 1)
      ;;
    one_step_no_condition)
      EXTRA_ARGS=(--mmp-fragment-growth-steps 1 --disable-structure-prompt-condition-guided-ranking)
      ;;
    *)
      echo "Unknown ablation mode: $mode"
      exit 2
      ;;
  esac

  echo "Submitting ablation: $mode"
  PHYSTABMOL_RUN_DIR="$RUN_DIR" \
  PHYSTABMOL_WALLTIME="$WALLTIME" \
  PHYSTABMOL_STRUCTURE_PROMPT_CONDITIONS="$CONDITIONS" \
  PHYSTABMOL_STRUCTURE_PROMPT_SAMPLES="$SAMPLES" \
  PHYSTABMOL_STRUCTURE_PROMPT_DECODE_TOP_K="$DECODE_TOP_K" \
    bash scripts/resume_structure_prompt_benchmark.sh \
      --ablation-name "$mode" \
      "${EXTRA_ARGS[@]}"
done

