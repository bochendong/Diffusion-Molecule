#!/usr/bin/env bash
# Submit a randomized-SMILES augmentation-strength sweep for Phase 5A-6.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SKETCHMOL_COMPARE_SEED:-${SKETCHSMILES_SEED:-7}}"
COUNTS="${SKETCHSMILES_RANDOMIZED_SWEEP_COUNTS:-1 2 4 8}"

echo "SketchMolCompare -> SketchSMILES Phase 5A-6 augmentation sweep"
echo "  seed=$SEED"
echo "  counts=$COUNTS"

for COUNT in $COUNTS; do
  echo
  echo "Submitting 5A-6 randomized_smiles_per_molecule=$COUNT"
  SKETCHSMILES_SEED="$SEED" \
  SKETCHSMILES_RUN_NAME="sketchmol_compare_phase5a6_aug${COUNT}_seed${SEED}" \
  SKETCHSMILES_RANDOMIZED_SMILES_PER_MOLECULE="$COUNT" \
  SKETCHSMILES_SLURM_JOB_NAME="sketchsmiles-5a6-a${COUNT}" \
  bash "$SCRIPT_DIR/submit_sketchsmiles_5a6_randomized.sh" "$@"
done
