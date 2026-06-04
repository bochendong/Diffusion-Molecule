#!/usr/bin/env bash
# Run the four publishable trajectory-memory baselines.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TRAJECTORY_PATH="${LETA_TRAJECTORY_PATH:-outputs/trajectories/sketchmol_opt_bootstrap.jsonl}"
SUITE_NAME="${LETA_SUITE_NAME:-sketchmol_trajectory_suite_seed${LETA_SEED:-7}}"

if [[ ! -f "$TRAJECTORY_PATH" ]]; then
  echo "Trajectory file not found, bootstrapping: $TRAJECTORY_PATH"
  LETA_TRAJECTORY_JSONL="$TRAJECTORY_PATH" bash "$SCRIPT_DIR/bootstrap_sketchmol_trajectories.sh"
fi

MODEL_KINDS=(history current_only no_reward_history shuffled_history)
for MODEL_KIND in "${MODEL_KINDS[@]}"; do
  echo
  echo "=== Running $MODEL_KIND ==="
  LETA_MODEL_KIND="$MODEL_KIND" \
  LETA_RUN_NAME="${SUITE_NAME}_${MODEL_KIND}" \
  bash "$SCRIPT_DIR/run_trajectory_training.sh"
done

echo
echo "Baseline suite finished: outputs/runs/${SUITE_NAME}_*"

