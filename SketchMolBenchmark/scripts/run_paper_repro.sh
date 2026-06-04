#!/usr/bin/env bash
# Run the SketchMol paper-reproduction pathway:
# condition -> SketchMol diffusion image -> MolScribe OCR -> benchmark artifacts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SKETCHMOL_REPO="${SKETCHMOL_REPO:-Research/Molecule Generation/SketchMol/SketchMol-v1-main}"
SKETCHMOL_PYTHON_BIN="${SKETCHMOL_PYTHON_BIN:-python}"
SKETCHMOL_MOLSCRIBE_PYTHON_BIN="${SKETCHMOL_MOLSCRIBE_PYTHON_BIN:-}"
SKETCHMOL_MOLSCRIBE_WORKDIR="${SKETCHMOL_MOLSCRIBE_WORKDIR:-$SKETCHMOL_REPO/evaluate}"
SKETCHMOL_MODULES="${SKETCHMOL_MODULES:-gcc opencv/4.13.0 rdkit/2024.09.6}"

SKETCHMOL_CKPT="${SKETCHMOL_CKPT:-}"
SKETCHMOL_MOLSCRIBE_MODEL="${SKETCHMOL_MOLSCRIBE_MODEL:-}"
SKETCHMOL_OUTPUT_ROOT="${SKETCHMOL_OUTPUT_ROOT:-$REPO_ROOT/SketchMolBenchmark/runs}"
SKETCHMOL_RUN_NAME="${SKETCHMOL_RUN_NAME:-paper_repro_mw400_$(date +%Y%m%d_%H%M%S)}"
SKETCHMOL_BENCHMARK_OUTPUT_DIR="${SKETCHMOL_BENCHMARK_OUTPUT_DIR:-SketchMolBenchmark/outputs/paper_repro}"

# Defaults copied from the SketchMol README MW:400 paper-style example.
SKETCHMOL_PRESET_STR="${SKETCHMOL_PRESET_STR:-MW:400}"
SKETCHMOL_POST="${SKETCHMOL_POST:-mw_400_${SKETCHMOL_RUN_NAME}}"
SKETCHMOL_CONDITIONAL_COUNT="${SKETCHMOL_CONDITIONAL_COUNT:-40}"
SKETCHMOL_N_SAMPLES="${SKETCHMOL_N_SAMPLES:-1}"
SKETCHMOL_CUSTOM_STEPS="${SKETCHMOL_CUSTOM_STEPS:-250}"
SKETCHMOL_SCALE="${SKETCHMOL_SCALE:-1.2}"
SKETCHMOL_SCALE_PRO="${SKETCHMOL_SCALE_PRO:-6.3}"
SKETCHMOL_PROPERTY_NUM="${SKETCHMOL_PROPERTY_NUM:-1}"
SKETCHMOL_TRI="${SKETCHMOL_TRI:-true}"
SKETCHMOL_MOLSCRIBE_BATCH_SIZE="${SKETCHMOL_MOLSCRIBE_BATCH_SIZE:-16}"

require_file() {
  local value="$1"
  local label="$2"
  if [[ -z "$value" || ! -f "$value" ]]; then
    echo "ERROR: $label must point to a real file: ${value:-<empty>}" >&2
    exit 2
  fi
}

require_executable() {
  local value="$1"
  local label="$2"
  if [[ -z "$value" ]]; then
    echo "ERROR: set $label to the intended Python executable." >&2
    exit 2
  fi
  if ! command -v "$value" >/dev/null 2>&1 && [[ ! -x "$value" ]]; then
    echo "ERROR: $label is not executable: $value" >&2
    exit 2
  fi
}

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    # Slurm --wrap shells may not initialize Lmod automatically.
    # shellcheck source=/dev/null
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  fi
fi

if command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_MODULES
fi

if [[ ! -d "$SKETCHMOL_REPO" ]]; then
  echo "ERROR: SketchMol repo not found: $SKETCHMOL_REPO" >&2
  exit 2
fi
if [[ ! -d "$SKETCHMOL_MOLSCRIBE_WORKDIR" ]]; then
  echo "ERROR: MolScribe workdir not found: $SKETCHMOL_MOLSCRIBE_WORKDIR" >&2
  exit 2
fi
SKETCHMOL_REPO_ABS="$(cd "$SKETCHMOL_REPO" && pwd)"
SKETCHMOL_MOLSCRIBE_WORKDIR_ABS="$(cd "$SKETCHMOL_MOLSCRIBE_WORKDIR" && pwd)"

require_executable "$SKETCHMOL_PYTHON_BIN" "SKETCHMOL_PYTHON_BIN"
require_executable "$SKETCHMOL_MOLSCRIBE_PYTHON_BIN" "SKETCHMOL_MOLSCRIBE_PYTHON_BIN"
require_file "$SKETCHMOL_CKPT" "SKETCHMOL_CKPT"
require_file "$SKETCHMOL_MOLSCRIBE_MODEL" "SKETCHMOL_MOLSCRIBE_MODEL"

RUN_DIR="$SKETCHMOL_OUTPUT_ROOT/$SKETCHMOL_RUN_NAME"
mkdir -p "$RUN_DIR"
RUN_LOG="$RUN_DIR/run.log"

echo "SketchMol paper reproduction benchmark" | tee "$RUN_LOG"
echo "  run_name=$SKETCHMOL_RUN_NAME" | tee -a "$RUN_LOG"
echo "  sketchmol_repo=$SKETCHMOL_REPO_ABS" | tee -a "$RUN_LOG"
echo "  sketchmol_python=$SKETCHMOL_PYTHON_BIN" | tee -a "$RUN_LOG"
echo "  molscribe_python=$SKETCHMOL_MOLSCRIBE_PYTHON_BIN" | tee -a "$RUN_LOG"
echo "  molscribe_workdir=$SKETCHMOL_MOLSCRIBE_WORKDIR_ABS" | tee -a "$RUN_LOG"
echo "  ckpt=$SKETCHMOL_CKPT" | tee -a "$RUN_LOG"
echo "  molscribe_model=$SKETCHMOL_MOLSCRIBE_MODEL" | tee -a "$RUN_LOG"
echo "  preset=$SKETCHMOL_PRESET_STR" | tee -a "$RUN_LOG"
echo "  scale=$SKETCHMOL_SCALE" | tee -a "$RUN_LOG"
echo "  scale_pro=$SKETCHMOL_SCALE_PRO" | tee -a "$RUN_LOG"
echo "  tri=$SKETCHMOL_TRI" | tee -a "$RUN_LOG"
echo "  conditional_count=$SKETCHMOL_CONDITIONAL_COUNT" | tee -a "$RUN_LOG"

pushd "$SKETCHMOL_REPO_ABS" >/dev/null
set +e
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
"$SKETCHMOL_PYTHON_BIN" scripts/sample_diffusion_condition_continuousV2.py \
  -r "$SKETCHMOL_CKPT" \
  --logdir "$RUN_DIR/sketchmol_logs" \
  --post "$SKETCHMOL_POST" \
  -p "$SKETCHMOL_PRESET_STR" \
  --conditional_count "$SKETCHMOL_CONDITIONAL_COUNT" \
  -n "$SKETCHMOL_N_SAMPLES" \
  -c "$SKETCHMOL_CUSTOM_STEPS" \
  --scale "$SKETCHMOL_SCALE" \
  --scale_pro "$SKETCHMOL_SCALE_PRO" \
  --property_num "$SKETCHMOL_PROPERTY_NUM" \
  --tri "$SKETCHMOL_TRI" \
  2>&1 | tee -a "$RUN_LOG"
SAMPLE_STATUS=${PIPESTATUS[0]}
set -e
popd >/dev/null

if [[ "$SAMPLE_STATUS" -ne 0 ]]; then
  echo "ERROR: SketchMol sampling failed with status $SAMPLE_STATUS" | tee -a "$RUN_LOG" >&2
  exit "$SAMPLE_STATUS"
fi

IMAGE_CSV="$(grep -Eo 'path save to .*/image_path.csv' "$RUN_LOG" | tail -1 | sed 's/^path save to //')"
if [[ -z "$IMAGE_CSV" || ! -f "$IMAGE_CSV" ]]; then
  IMAGE_CSV="$(find "$RUN_DIR/sketchmol_logs" -name image_path.csv -type f | sort | tail -1)"
fi
if [[ -z "$IMAGE_CSV" || ! -f "$IMAGE_CSV" ]]; then
  echo "ERROR: could not find SketchMol image_path.csv under $RUN_DIR" | tee -a "$RUN_LOG" >&2
  exit 2
fi
echo "  image_csv=$IMAGE_CSV" | tee -a "$RUN_LOG"

pushd "$SKETCHMOL_MOLSCRIBE_WORKDIR_ABS" >/dev/null
set +e
PYTHONPATH="$PWD:$SKETCHMOL_REPO_ABS/evaluate:$SKETCHMOL_REPO_ABS${PYTHONPATH:+:$PYTHONPATH}" \
"$SKETCHMOL_MOLSCRIBE_PYTHON_BIN" predict_csv.py \
  --model_path "$SKETCHMOL_MOLSCRIBE_MODEL" \
  --image_path "$IMAGE_CSV" \
  -n "$SKETCHMOL_MOLSCRIBE_BATCH_SIZE" \
  2>&1 | tee -a "$RUN_LOG"
OCR_STATUS=${PIPESTATUS[0]}
set -e
popd >/dev/null

if [[ "$OCR_STATUS" -ne 0 ]]; then
  echo "ERROR: MolScribe OCR failed with status $OCR_STATUS" | tee -a "$RUN_LOG" >&2
  exit "$OCR_STATUS"
fi

SKETCHMOL_BENCHMARK_SOURCE_CSV="$IMAGE_CSV" \
SKETCHMOL_BENCHMARK_RUN_LOG="$RUN_LOG" \
SKETCHMOL_BENCHMARK_OUTPUT_DIR="$SKETCHMOL_BENCHMARK_OUTPUT_DIR" \
SKETCHMOL_BENCHMARK_NAME="$SKETCHMOL_RUN_NAME" \
SKETCHMOL_REPO="$SKETCHMOL_REPO_ABS" \
SKETCHMOL_BENCHMARK_PYTHON_BIN="${SKETCHMOL_BENCHMARK_PYTHON_BIN:-$SKETCHMOL_MOLSCRIBE_PYTHON_BIN}" \
bash "$PROJECT_DIR/scripts/materialize_current.sh" | tee -a "$RUN_LOG"

echo
echo "SketchMol paper reproduction finished: $SKETCHMOL_BENCHMARK_OUTPUT_DIR"
echo "  image_csv=$IMAGE_CSV"
echo "  log=$RUN_LOG"
echo "  metrics=$SKETCHMOL_BENCHMARK_OUTPUT_DIR/metrics.json"
