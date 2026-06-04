#!/usr/bin/env bash
# Submit current-state-only SketchMol opt-pair baseline to Slurm.

set -euo pipefail

export LETA_MODEL_KIND="${LETA_MODEL_KIND:-current_only}"
export LETA_RUN_NAME="${LETA_RUN_NAME:-sketchmol_opt_pairs_current_only_seed${LETA_SEED:-7}}"
export LETA_SLURM_JOB_NAME="${LETA_SLURM_JOB_NAME:-leta-current-only}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/submit_sketchmol_opt_training.sh"

