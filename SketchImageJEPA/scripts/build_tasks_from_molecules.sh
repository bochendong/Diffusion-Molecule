#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
MOLECULE_CSV="${SKETCHIMAGE_MOLECULE_CSV:-${1:-}}"
OUT="${SKETCHIMAGE_TASK_CSV:-data/generated_tasks.csv}"

if [[ -z "$MOLECULE_CSV" ]]; then
  echo "Usage: SKETCHIMAGE_MOLECULE_CSV=/path/to/molecules.csv bash scripts/build_tasks_from_molecules.sh" >&2
  echo "   or: bash scripts/build_tasks_from_molecules.sh /path/to/molecules.csv" >&2
  exit 2
fi

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m sketchimage_jepa.task_builder \
  --molecule-csv "$MOLECULE_CSV" \
  --out "$OUT" \
  --limit "${SKETCHIMAGE_MOLECULE_LIMIT:-10000}" \
  --max-tasks "${SKETCHIMAGE_MAX_TASKS:-5000}" \
  --pairs-per-source "${SKETCHIMAGE_PAIRS_PER_SOURCE:-2}" \
  --pair-candidates "${SKETCHIMAGE_PAIR_CANDIDATES:-128}" \
  --min-similarity "${SKETCHIMAGE_MIN_SIMILARITY:-0.15}" \
  --max-similarity "${SKETCHIMAGE_MAX_SIMILARITY:-0.90}" \
  --seed "${SKETCHIMAGE_SEED:-7}" \
  --task-types "${SKETCHIMAGE_TASK_TYPES:-de_novo,edit,inpaint,fragment_grow}"
