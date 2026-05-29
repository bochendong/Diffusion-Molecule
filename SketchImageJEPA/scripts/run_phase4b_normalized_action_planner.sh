#!/usr/bin/env bash
# Run Phase 4B: normalized edit/action planner.

set -euo pipefail

export SKETCHIMAGE_RUN_NAME="${SKETCHIMAGE_RUN_NAME:-phase4b_normalized_action_planner_2048_seed${SKETCHIMAGE_SEED:-7}}"
export SKETCHIMAGE_PHASE_TITLE="${SKETCHIMAGE_PHASE_TITLE:-Phase 4B normalized edit/action planner}"
export SKETCHIMAGE_ACTION_TARGET_MODE="${SKETCHIMAGE_ACTION_TARGET_MODE:-unit_direction}"
export SKETCHIMAGE_ACTION_STEP_MODE="${SKETCHIMAGE_ACTION_STEP_MODE:-target_norm_median}"
export SKETCHIMAGE_ACTION_ALPHAS="${SKETCHIMAGE_ACTION_ALPHAS:-0.05,0.10,0.15,0.25,0.50,0.75,1.00}"
export SKETCHIMAGE_ALPHA_SCORE_PENALTY="${SKETCHIMAGE_ALPHA_SCORE_PENALTY:-0.01}"

bash "$(dirname "${BASH_SOURCE[0]}")/run_phase4_edit_action_planner.sh"
