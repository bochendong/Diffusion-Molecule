#!/usr/bin/env bash
# Audit whether a completed run or explicit split can be solved by train-set shortcuts.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
RUN_DIR="${1:-${SKETCHIMAGE_RUN_DIR:-}}"

if [[ -n "${SKETCHIMAGE_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHIMAGE_MODULES
fi

if [[ -n "$RUN_DIR" ]]; then
  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m sketchimage_jepa.benchmark_audit --run-dir "$RUN_DIR"
else
  if [[ -z "${SKETCHIMAGE_TRAIN_CSV:-}" || -z "${SKETCHIMAGE_EVAL_CSV:-}" ]]; then
    echo "Provide a run directory, or set SKETCHIMAGE_TRAIN_CSV and SKETCHIMAGE_EVAL_CSV." >&2
    exit 2
  fi
  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m sketchimage_jepa.benchmark_audit \
    --train-csv "$SKETCHIMAGE_TRAIN_CSV" \
    --eval-csv "$SKETCHIMAGE_EVAL_CSV" \
    --predictions-csv "${SKETCHIMAGE_PREDICTIONS_CSV:-}" \
    --out-dir "${SKETCHIMAGE_AUDIT_OUT_DIR:-outputs/audit}"
fi
