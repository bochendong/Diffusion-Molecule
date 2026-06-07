#!/usr/bin/env bash
# Continue the real SketchMol benchmark from an existing image_path.csv.
# This avoids rerunning diffusion sampling when only MolScribe/OCR failed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../SketchMol-Understanding-Condition/scripts/molscribe_env.sh
source "$REPO_ROOT/SketchMol-Understanding-Condition/scripts/molscribe_env.sh"

SOURCE_CSV="${SKETCHMOL_BENCHMARK_SOURCE_CSV:-${1:-}}"
SKETCHMOL_REPO="${SKETCHMOL_REPO:-$SKETCHMOL_ROOT}"
PYTHON_BIN="${SKETCHMOL_MOLSCRIBE_PYTHON_BIN:-${SKETCHMOL_PYTHON_BIN:-python}}"
BATCH_SIZE="${SKETCHMOL_MOLSCRIBE_BATCH_SIZE:-100}"
OUTPUT_DIR="${SKETCHMOL_BENCHMARK_OUTPUT_DIR:-SketchMolBenchmark/outputs/current}"
SKETCHMOL_MODULES="${SKETCHMOL_MODULES:-gcc opencv/4.13.0 rdkit/2024.09.6}"

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    # shellcheck source=/dev/null
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  fi
fi

if command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_MODULES
fi

if [[ -z "$SOURCE_CSV" ]]; then
  cat <<EOF >&2
ERROR: provide the existing SketchMol image_path.csv.

Example:
  SKETCHMOL_BENCHMARK_SOURCE_CSV=/path/to/image_path.csv \\
  SKETCHMOL_MOLSCRIBE_MODEL=/path/to/swin_base_char_aux_200k.pth \\
  SKETCHMOL_MOLSCRIBE_PYTHON_BIN=/home/bdong/.venvs/molscribe_overlay/bin/python \\
  bash SketchMolBenchmark/scripts/resume_real_sketchmol_ocr.sh
EOF
  exit 2
fi

if [[ ! -f "$SOURCE_CSV" ]]; then
  echo "ERROR: source CSV does not exist: $SOURCE_CSV" >&2
  exit 2
fi
SOURCE_CSV="$(cd "$(dirname "$SOURCE_CSV")" && pwd)/$(basename "$SOURCE_CSV")"

if [[ ! -d "$SKETCHMOL_REPO" ]]; then
  echo "ERROR: SketchMol repo not found: $SKETCHMOL_REPO" >&2
  exit 2
fi

if [[ -z "${SKETCHMOL_MOLSCRIBE_MODEL:-}" || ! -f "$SKETCHMOL_MOLSCRIBE_MODEL" ]]; then
  echo "ERROR: SKETCHMOL_MOLSCRIBE_MODEL must point to a MolScribe checkpoint." >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

INFERRED_RUN_NAME="$(printf '%s\n' "$SOURCE_CSV" | sed -n 's#.*SketchMolBenchmark/runs/\([^/]*\)/.*#\1#p')"
BENCHMARK_NAME="${SKETCHMOL_RUN_NAME:-${SKETCHMOL_BENCHMARK_NAME:-${INFERRED_RUN_NAME:-real_sketchmol_ocr_resume_$(date +%Y%m%d_%H%M%S)}}}"
RUN_DIR="SketchMolBenchmark/runs/$BENCHMARK_NAME"
mkdir -p "$RUN_DIR"
RUN_LOG="${SKETCHMOL_BENCHMARK_RUN_LOG:-$RUN_DIR/resume_ocr.log}"
if [[ "$RUN_LOG" = /* ]]; then
  RUN_LOG_ABS="$RUN_LOG"
else
  RUN_LOG_ABS="$REPO_ROOT/$RUN_LOG"
fi
mkdir -p "$(dirname "$RUN_LOG_ABS")"

echo "Resume real SketchMol + OCR benchmark:"
echo "  python=$PYTHON_BIN"
echo "  onmt_overlay=$ONMT_OVERLAY"
echo "  molscribe_workdir=$MOLSCRIBE_WORKDIR"
echo "  sketchmol_repo=$SKETCHMOL_REPO"
echo "  molscribe_model=$SKETCHMOL_MOLSCRIBE_MODEL"
echo "  image_csv=$SOURCE_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  benchmark_name=$BENCHMARK_NAME"

prepend_molscribe_pythonpath
PYTHON_BIN="$PYTHON_BIN" check_molscribe_import

pushd "$SKETCHMOL_REPO" >/dev/null
"$PYTHON_BIN" evaluate/predict_csv.py \
  --model_path "$SKETCHMOL_MOLSCRIBE_MODEL" \
  --image_path "$SOURCE_CSV" \
  -n "$BATCH_SIZE" \
  2>&1 | tee -a "$RUN_LOG_ABS"
popd >/dev/null

SKETCHMOL_BENCHMARK_SOURCE_CSV="$SOURCE_CSV" \
SKETCHMOL_BENCHMARK_RUN_LOG="$RUN_LOG_ABS" \
SKETCHMOL_BENCHMARK_OUTPUT_DIR="$OUTPUT_DIR" \
SKETCHMOL_BENCHMARK_NAME="$BENCHMARK_NAME" \
SKETCHMOL_REPO="$SKETCHMOL_REPO" \
SKETCHMOL_BENCHMARK_PYTHON_BIN="$PYTHON_BIN" \
bash "$PROJECT_DIR/scripts/materialize_current.sh" | tee -a "$RUN_LOG_ABS"

echo
echo "Real SketchMol + OCR benchmark resumed: $OUTPUT_DIR"
echo "  image_csv=$SOURCE_CSV"
echo "  log=$RUN_LOG_ABS"
echo "  metrics=$OUTPUT_DIR/metrics.json"
