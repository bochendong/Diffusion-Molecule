#!/usr/bin/env bash
# Execute one deterministic controller reconciliation after an experiment ends.

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 PLAN_JSON STATE_JSON" >&2
  exit 2
fi

PLAN_JSON="$1"
STATE_JSON="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export SUCC_UCA_AUTOMATION_REPO_DIR="${SUCC_UCA_AUTOMATION_REPO_DIR:-$REPO_DIR}"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
fi

PYTHON_BIN="${SUCC_UCA_AUTOMATION_PYTHON_BIN:-python3}"
"$PYTHON_BIN" "$SCRIPT_DIR/experiment_loop.py" \
  --plan "$PLAN_JSON" \
  --state "$STATE_JSON" \
  reconcile
