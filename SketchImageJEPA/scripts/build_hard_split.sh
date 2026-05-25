#!/usr/bin/env bash
# Build hard train/eval splits that hold out target scaffolds and near neighbors.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SPLIT_NAME="${SKETCHIMAGE_HARD_SPLIT_NAME:-sketchmol_hard_seed${SKETCHIMAGE_SEED:-7}}"
TASK_CSV="${SKETCHIMAGE_TASK_CSV:-outputs/tasks/${SPLIT_NAME}_tasks.csv}"
TRAIN_OUT="${SKETCHIMAGE_TRAIN_OUT:-outputs/tasks/${SPLIT_NAME}_train.csv}"
EVAL_OUT="${SKETCHIMAGE_EVAL_OUT:-outputs/tasks/${SPLIT_NAME}_eval.csv}"
SUMMARY_OUT="${SKETCHIMAGE_SUMMARY_OUT:-outputs/tasks/${SPLIT_NAME}_summary.json}"
SEED="${SKETCHIMAGE_SEED:-7}"
EVAL_FRACTION="${SKETCHIMAGE_EVAL_FRACTION:-0.2}"
MAX_TANIMOTO="${SKETCHIMAGE_MAX_TRAIN_TARGET_TANIMOTO:-0.55}"

if [[ -n "${SKETCHIMAGE_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHIMAGE_MODULES
fi

if [[ -n "${SKETCHIMAGE_MOLECULE_CSV:-}" && ! -f "$TASK_CSV" ]]; then
  echo "Building task CSV before hard split:"
  echo "  task_csv=$TASK_CSV"
  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m sketchimage_jepa.task_builder \
    --molecule-csv "$SKETCHIMAGE_MOLECULE_CSV" \
    --out "$TASK_CSV" \
    --limit "${SKETCHIMAGE_MOLECULE_LIMIT:-50000}" \
    --max-tasks "${SKETCHIMAGE_MAX_TASKS:-10000}" \
    --pairs-per-source "${SKETCHIMAGE_PAIRS_PER_SOURCE:-2}" \
    --pair-candidates "${SKETCHIMAGE_PAIR_CANDIDATES:-128}" \
    --min-similarity "${SKETCHIMAGE_MIN_SIMILARITY:-0.15}" \
    --max-similarity "${SKETCHIMAGE_MAX_SIMILARITY:-0.90}" \
    --seed "$SEED" \
    --task-types "${SKETCHIMAGE_TASK_TYPES:-de_novo,edit,inpaint,fragment_grow}"
fi

if [[ ! -f "$TASK_CSV" ]]; then
  echo "ERROR: task CSV does not exist: $TASK_CSV" >&2
  echo "Set SKETCHIMAGE_TASK_CSV, or set SKETCHIMAGE_MOLECULE_CSV so this script can build one." >&2
  exit 2
fi

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m sketchimage_jepa.hard_split \
  --task-csv "$TASK_CSV" \
  --train-out "$TRAIN_OUT" \
  --eval-out "$EVAL_OUT" \
  --summary-out "$SUMMARY_OUT" \
  --eval-fraction "$EVAL_FRACTION" \
  --seed "$SEED" \
  --max-train-target-tanimoto "$MAX_TANIMOTO"

echo
echo "Hard split written:"
echo "  train_csv=$TRAIN_OUT"
echo "  eval_csv=$EVAL_OUT"
echo "  summary=$SUMMARY_OUT"
