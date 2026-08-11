#!/usr/bin/env bash
# Start the bounded experiment loop from a Nibi login node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PLAN_JSON="${SUCC_UCA_AUTOMATION_PLAN:-$SCRIPT_DIR/experiment_plan.json}"
ROUND_ID="${1:-retrieved_delta_oracle_repair_v5}"
export SUCC_UCA_AUTOMATION_REPO_DIR="${SUCC_UCA_AUTOMATION_REPO_DIR:-$REPO_DIR}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on a Slurm login node." >&2
  exit 2
fi

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
fi

PYTHON_BIN="${SUCC_UCA_AUTOMATION_PYTHON_BIN:-python3}"
"$PYTHON_BIN" "$SCRIPT_DIR/experiment_loop.py" --plan "$PLAN_JSON" validate
"$PYTHON_BIN" "$SCRIPT_DIR/experiment_loop.py" --plan "$PLAN_JSON" start --round "$ROUND_ID"
