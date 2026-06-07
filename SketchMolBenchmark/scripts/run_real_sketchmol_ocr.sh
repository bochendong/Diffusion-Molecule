#!/usr/bin/env bash
# Run the real SketchMol image-generation + MolScribe/OCR pipeline, then
# materialize its output into SketchMolBenchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SKETCHMOL_REPO="${SKETCHMOL_REPO:-Research/Molecule Generation/SketchMol/SketchMol-v1-main}"
SKETCHMOL_PYTHON_BIN="${SKETCHMOL_PYTHON_BIN:-python}"
SKETCHMOL_MOLSCRIBE_PYTHON_BIN="${SKETCHMOL_MOLSCRIBE_PYTHON_BIN:-$SKETCHMOL_PYTHON_BIN}"
SKETCHMOL_CKPT="${SKETCHMOL_CKPT:-}"
SKETCHMOL_MOLSCRIBE_MODEL="${SKETCHMOL_MOLSCRIBE_MODEL:-}"
SKETCHMOL_OUTPUT_ROOT="${SKETCHMOL_OUTPUT_ROOT:-$REPO_ROOT/SketchMolBenchmark/runs}"
SKETCHMOL_RUN_NAME="${SKETCHMOL_RUN_NAME:-real_sketchmol_ocr_$(date +%Y%m%d_%H%M%S)}"
SKETCHMOL_PRESET_STR="${SKETCHMOL_PRESET_STR:-MW:400}"
SKETCHMOL_POST="${SKETCHMOL_POST:-_${SKETCHMOL_RUN_NAME}}"
SKETCHMOL_CONDITIONAL_COUNT="${SKETCHMOL_CONDITIONAL_COUNT:-8}"
SKETCHMOL_N_SAMPLES="${SKETCHMOL_N_SAMPLES:-1}"
SKETCHMOL_CUSTOM_STEPS="${SKETCHMOL_CUSTOM_STEPS:-250}"
SKETCHMOL_SCALE="${SKETCHMOL_SCALE:-2}"
SKETCHMOL_SCALE_PRO="${SKETCHMOL_SCALE_PRO:-4}"
SKETCHMOL_PROPERTY_NUM="${SKETCHMOL_PROPERTY_NUM:-1}"
SKETCHMOL_TRI="${SKETCHMOL_TRI:-true}"
SKETCHMOL_BENCHMARK_OUTPUT_DIR="${SKETCHMOL_BENCHMARK_OUTPUT_DIR:-SketchMolBenchmark/outputs/current}"

check_python_imports() {
  local label="$1"
  local python_bin="$2"
  shift 2

  if ! command -v "$python_bin" >/dev/null 2>&1 && [[ ! -x "$python_bin" ]]; then
    echo "ERROR: $label Python executable not found: $python_bin" >&2
    exit 2
  fi

  if ! "$python_bin" - "$@" <<'PY'
import importlib
import sys

missing = []
for module_name in sys.argv[1:]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        missing.append((module_name, exc))

if missing:
    print("ERROR: Python environment is missing required modules:", file=sys.stderr)
    for module_name, exc in missing:
        print(f"  {module_name}: {exc}", file=sys.stderr)
    sys.exit(2)
PY
  then
    echo "       Install the missing modules into this interpreter before rerunning:" >&2
    echo "       $python_bin -m pip install <missing-module>" >&2
    exit 2
  fi
}

if [[ ! -d "$SKETCHMOL_REPO" ]]; then
  echo "ERROR: real SketchMol repo not found: $SKETCHMOL_REPO" >&2
  exit 2
fi
if [[ -z "$SKETCHMOL_CKPT" || ! -f "$SKETCHMOL_CKPT" ]]; then
  echo "ERROR: SKETCHMOL_CKPT must point to a real SketchMol diffusion checkpoint." >&2
  exit 2
fi
if [[ -z "$SKETCHMOL_MOLSCRIBE_MODEL" || ! -f "$SKETCHMOL_MOLSCRIBE_MODEL" ]]; then
  echo "ERROR: SKETCHMOL_MOLSCRIBE_MODEL must point to a MolScribe checkpoint." >&2
  exit 2
fi

export PYTHONPATH="$SKETCHMOL_REPO${PYTHONPATH:+:$PYTHONPATH}"

check_python_imports \
  "SketchMol" \
  "$SKETCHMOL_PYTHON_BIN" \
  yaml torch numpy pandas tqdm omegaconf PIL einops ldm.data.pubchemdata ldm.models.diffusion.ddpm
check_python_imports \
  "MolScribe" \
  "$SKETCHMOL_MOLSCRIBE_PYTHON_BIN" \
  torch numpy pandas PIL cv2 albumentations

RUN_DIR="$SKETCHMOL_OUTPUT_ROOT/$SKETCHMOL_RUN_NAME"
mkdir -p "$RUN_DIR"
RUN_LOG="$RUN_DIR/run.log"

echo "Real SketchMol + OCR benchmark" | tee "$RUN_LOG"
echo "  sketchmol_repo=$SKETCHMOL_REPO" | tee -a "$RUN_LOG"
echo "  sketchmol_python=$SKETCHMOL_PYTHON_BIN" | tee -a "$RUN_LOG"
echo "  molscribe_python=$SKETCHMOL_MOLSCRIBE_PYTHON_BIN" | tee -a "$RUN_LOG"
echo "  ckpt=$SKETCHMOL_CKPT" | tee -a "$RUN_LOG"
echo "  molscribe_model=$SKETCHMOL_MOLSCRIBE_MODEL" | tee -a "$RUN_LOG"
echo "  preset=$SKETCHMOL_PRESET_STR" | tee -a "$RUN_LOG"
echo "  run_name=$SKETCHMOL_RUN_NAME" | tee -a "$RUN_LOG"

pushd "$SKETCHMOL_REPO" >/dev/null

set +e
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

# shellcheck source=../../SketchMol-Understanding-Condition/scripts/molscribe_env.sh
source "$REPO_ROOT/SketchMol-Understanding-Condition/scripts/molscribe_env.sh"
prepend_molscribe_pythonpath
PYTHON_BIN="$SKETCHMOL_MOLSCRIBE_PYTHON_BIN" check_molscribe_import

set +e
"$SKETCHMOL_MOLSCRIBE_PYTHON_BIN" evaluate/predict_csv.py \
  --model_path "$SKETCHMOL_MOLSCRIBE_MODEL" \
  --image_path "$IMAGE_CSV" \
  -n "${SKETCHMOL_MOLSCRIBE_BATCH_SIZE:-100}" \
  2>&1 | tee -a "$RUN_LOG"
OCR_STATUS=${PIPESTATUS[0]}
set -e

if [[ "$OCR_STATUS" -ne 0 ]]; then
  echo "ERROR: MolScribe OCR failed with status $OCR_STATUS" | tee -a "$RUN_LOG" >&2
  exit "$OCR_STATUS"
fi

popd >/dev/null

SKETCHMOL_BENCHMARK_SOURCE_CSV="$IMAGE_CSV" \
SKETCHMOL_BENCHMARK_RUN_LOG="$RUN_LOG" \
SKETCHMOL_BENCHMARK_OUTPUT_DIR="$SKETCHMOL_BENCHMARK_OUTPUT_DIR" \
SKETCHMOL_BENCHMARK_NAME="$SKETCHMOL_RUN_NAME" \
SKETCHMOL_REPO="$SKETCHMOL_REPO" \
SKETCHMOL_BENCHMARK_PYTHON_BIN="${SKETCHMOL_BENCHMARK_PYTHON_BIN:-$SKETCHMOL_MOLSCRIBE_PYTHON_BIN}" \
bash "$PROJECT_DIR/scripts/materialize_current.sh" | tee -a "$RUN_LOG"

echo
echo "Real SketchMol + OCR benchmark finished: $SKETCHMOL_BENCHMARK_OUTPUT_DIR"
echo "  image_csv=$IMAGE_CSV"
echo "  log=$RUN_LOG"
echo "  metrics=$SKETCHMOL_BENCHMARK_OUTPUT_DIR/metrics.json"
