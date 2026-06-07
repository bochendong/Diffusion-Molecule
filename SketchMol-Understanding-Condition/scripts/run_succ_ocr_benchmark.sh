#!/usr/bin/env bash
# Re-run MolScribe OCR + image-to-structure benchmark on existing UniVideo outputs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

# shellcheck source=./molscribe_env.sh
source "$SCRIPT_DIR/molscribe_env.sh"

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

MOLSCRIBE_BATCH_SIZE="${SUCC_MOLSCRIBE_BATCH_SIZE:-100}"
MOLSCRIBE_DEVICE="${SUCC_MOLSCRIBE_DEVICE:-cuda}"
MOLSCRIBE_BACKEND="${SUCC_MOLSCRIBE_BACKEND:-sketchmol}"
PREPROCESS_IMAGES="${SUCC_PREPROCESS_IMAGES:-0}"
RAW_SMILES_FALLBACK="${SUCC_RAW_SMILES_FALLBACK:-0}"
SOURCE_TANIMOTO_THRESHOLDS="${SUCC_SOURCE_TANIMOTO_THRESHOLDS:-0.4,0.6,0.8}"
DIAG_LIMIT="${SUCC_MOLSCRIBE_DIAG_LIMIT:-128}"
RUN_DIAGNOSTIC="${SUCC_RUN_MOLSCRIBE_DIAGNOSTIC:-1}"

if [[ ! -f "$IMAGE_CSV" ]]; then
  echo "ERROR: image CSV not found: $IMAGE_CSV" >&2
  exit 2
fi

OCR_ARGS=(
  --model-path "$MOLSCRIBE_MODEL"
  --image-csv "$IMAGE_CSV"
  --batch-size "$MOLSCRIBE_BATCH_SIZE"
  --device "$MOLSCRIBE_DEVICE"
  --backend "$MOLSCRIBE_BACKEND"
)
if [[ "$PREPROCESS_IMAGES" == "1" ]]; then
  OCR_ARGS+=(--preprocess-images)
else
  OCR_ARGS+=(--no-preprocess-images)
fi
if [[ "$MOLSCRIBE_BACKEND" == "custom" && "$RAW_SMILES_FALLBACK" == "1" ]]; then
  OCR_ARGS+=(--raw-smiles-fallback)
fi

DIAG_ARGS=(
  --run-dir "$UNIFIED_OUTPUT_DIR"
  --model-path "$MOLSCRIBE_MODEL"
  --limit "$DIAG_LIMIT"
  --batch-size "$MOLSCRIBE_BATCH_SIZE"
  --device "$MOLSCRIBE_DEVICE"
  --backend "$MOLSCRIBE_BACKEND"
)
if [[ "$PREPROCESS_IMAGES" == "1" ]]; then
  DIAG_ARGS+=(--preprocess-images)
else
  DIAG_ARGS+=(--no-preprocess-images)
fi

echo "SUCC OCR benchmark rerun"
echo "  python=$PYTHON_BIN"
echo "  molscribe_workdir=$MOLSCRIBE_WORKDIR"
echo "  onmt_overlay=$ONMT_OVERLAY"
echo "  run_dir=$UNIFIED_OUTPUT_DIR"
echo "  image_csv=$IMAGE_CSV"
echo "  benchmark_dir=$STRUCTURE_BENCHMARK_DIR"
echo "  molscribe_model=$MOLSCRIBE_MODEL"
echo "  molscribe_backend=$MOLSCRIBE_BACKEND"
echo "  molscribe_device=$MOLSCRIBE_DEVICE"
echo "  preprocess_images=$PREPROCESS_IMAGES"
echo "  raw_smiles_fallback=$RAW_SMILES_FALLBACK"

prepend_molscribe_pythonpath
check_molscribe_import

echo "Running MolScribe OCR"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/run_molscribe_ocr.py" "${OCR_ARGS[@]}"

if [[ "$RUN_DIAGNOSTIC" == "1" ]]; then
  echo "Running MolScribe diagnostic"
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/diagnose_univideo_molscribe_ocr.py" "${DIAG_ARGS[@]}"
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
