#!/usr/bin/env bash
# Re-run MolScribe OCR + image-to-structure benchmark on existing UniVideo outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink}"
STRUCTURE_BENCHMARK_DIR="${SUCC_STRUCTURE_BENCHMARK_DIR:-$UNIFIED_OUTPUT_DIR/univideo_molecule/image_structure_benchmark}"
IMAGE_CSV="${SUCC_IMAGE_CSV:-$STRUCTURE_BENCHMARK_DIR/image_path.csv}"
SKETCHMOL_ROOT="${SUCC_SKETCHMOL_ROOT:-Research/Molecule Generation/SketchMol/SketchMol-v1-main}"
DEFAULT_MOLSCRIBE_MODEL="/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth"

if [[ -n "${SUCC_MOLSCRIBE_MODEL:-}" ]]; then
  MOLSCRIBE_MODEL="$SUCC_MOLSCRIBE_MODEL"
elif [[ -n "${SKETCHMOL_MOLSCRIBE_MODEL:-}" ]]; then
  MOLSCRIBE_MODEL="$SKETCHMOL_MOLSCRIBE_MODEL"
elif [[ -f "$DEFAULT_MOLSCRIBE_MODEL" ]]; then
  MOLSCRIBE_MODEL="$DEFAULT_MOLSCRIBE_MODEL"
else
  echo "ERROR: MolScribe model missing. Set SUCC_MOLSCRIBE_MODEL." >&2
  exit 2
fi

MOLSCRIBE_WORKDIR="${SUCC_MOLSCRIBE_WORKDIR:-${SKETCHMOL_MOLSCRIBE_WORKDIR:-$SKETCHMOL_ROOT/evaluate}}"
MOLSCRIBE_BATCH_SIZE="${SUCC_MOLSCRIBE_BATCH_SIZE:-100}"
MOLSCRIBE_DEVICE="${SUCC_MOLSCRIBE_DEVICE:-cuda}"
MOLSCRIBE_BACKEND="${SUCC_MOLSCRIBE_BACKEND:-sketchmol}"
SOURCE_TANIMOTO_THRESHOLDS="${SUCC_SOURCE_TANIMOTO_THRESHOLDS:-0.4,0.6,0.8}"
DIAG_LIMIT="${SUCC_MOLSCRIBE_DIAG_LIMIT:-128}"
RUN_DIAGNOSTIC="${SUCC_RUN_MOLSCRIBE_DIAGNOSTIC:-1}"

prepend_molscribe_pythonpath() {
  if [[ -z "$MOLSCRIBE_WORKDIR" ]]; then
    return
  fi
  if [[ -d "$MOLSCRIBE_WORKDIR/evaluate/molscribe" ]]; then
    export PYTHONPATH="$MOLSCRIBE_WORKDIR/evaluate:$MOLSCRIBE_WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
  else
    export PYTHONPATH="$MOLSCRIBE_WORKDIR${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

if [[ ! -f "$IMAGE_CSV" ]]; then
  echo "ERROR: image CSV not found: $IMAGE_CSV" >&2
  exit 2
fi

echo "SUCC OCR benchmark rerun"
echo "  python=$PYTHON_BIN"
echo "  run_dir=$UNIFIED_OUTPUT_DIR"
echo "  image_csv=$IMAGE_CSV"
echo "  benchmark_dir=$STRUCTURE_BENCHMARK_DIR"
echo "  molscribe_model=$MOLSCRIBE_MODEL"
echo "  molscribe_backend=$MOLSCRIBE_BACKEND"
echo "  molscribe_device=$MOLSCRIBE_DEVICE"
echo "  preprocess_images=1"
echo "  raw_smiles_fallback=0"

prepend_molscribe_pythonpath
"$PYTHON_BIN" - <<'PY'
from molscribe import MolScribe  # noqa: F401

print("molscribe import ok")
PY

echo "Running MolScribe OCR"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/run_molscribe_ocr.py" \
  --model-path "$MOLSCRIBE_MODEL" \
  --image-csv "$IMAGE_CSV" \
  --batch-size "$MOLSCRIBE_BATCH_SIZE" \
  --device "$MOLSCRIBE_DEVICE" \
  --backend "$MOLSCRIBE_BACKEND" \
  --preprocess-images \
  --no-raw-smiles-fallback

if [[ "$RUN_DIAGNOSTIC" == "1" ]]; then
  echo "Running MolScribe diagnostic"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/diagnose_univideo_molscribe_ocr.py" \
    --run-dir "$UNIFIED_OUTPUT_DIR" \
    --model-path "$MOLSCRIBE_MODEL" \
    --limit "$DIAG_LIMIT" \
    --batch-size "$MOLSCRIBE_BATCH_SIZE" \
    --device "$MOLSCRIBE_DEVICE" \
    --backend "$MOLSCRIBE_BACKEND" \
    --preprocess-images
fi

echo "Running image-to-structure benchmark"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_univideo_image_benchmark.py" \
  --image-csv "$IMAGE_CSV" \
  --output-dir "$STRUCTURE_BENCHMARK_DIR" \
  --method "univideo_${SUCC_LATENT_BACKEND:-image_vae}" \
  --source-tanimoto-thresholds "$SOURCE_TANIMOTO_THRESHOLDS"

echo
echo "SUCC OCR benchmark finished:"
echo "  image_csv=$IMAGE_CSV"
echo "  diagnostic=$STRUCTURE_BENCHMARK_DIR/molscribe_diagnostic.md"
echo "  report=$STRUCTURE_BENCHMARK_DIR/benchmark_report.md"
echo "  summary=$STRUCTURE_BENCHMARK_DIR/benchmark_summary.csv"
