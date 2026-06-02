#!/usr/bin/env bash
# Submit a wider-beam Phase 5A-4 candidate coverage run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${SKETCHMOL_COMPARE_SEED:-${SKETCHSMILES_SEED:-7}}"

export SKETCHSMILES_SEED="$SEED"
export SKETCHSMILES_RUN_NAME="${SKETCHSMILES_RUN_NAME:-sketchmol_compare_phase5a4_beam32_seed${SEED}}"
export SKETCHSMILES_BEAM_SIZE="${SKETCHSMILES_BEAM_SIZE:-32}"
export SKETCHSMILES_SAMPLES_PER_CONDITION="${SKETCHSMILES_SAMPLES_PER_CONDITION:-8}"
export SKETCHSMILES_SLURM_JOB_NAME="${SKETCHSMILES_SLURM_JOB_NAME:-sketchsmiles-5a4-b32}"
export SKETCHSMILES_SLURM_TIME="${SKETCHSMILES_SLURM_TIME:-12:00:00}"

echo "SketchMolCompare -> SketchSMILES Phase 5A-4 beam32"
echo "  run_name=$SKETCHSMILES_RUN_NAME"
echo "  beam_size=$SKETCHSMILES_BEAM_SIZE"

bash "$SCRIPT_DIR/submit_sketchsmiles_5a4_transformer.sh" "$@"
