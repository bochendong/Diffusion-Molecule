#!/usr/bin/env bash
# Run Phase 5A-1 oracle-conditioned learned SMILES decoder.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PAIR_DIR="${SKETCHSMILES_PAIR_DIR:-outputs/pairs/phys_50k}"
RUN_NAME="${SKETCHSMILES_RUN_NAME:-phase5a1_learned_smiles_decoder_seed${SKETCHSMILES_SEED:-7}}"
OUTPUT_DIR="${SKETCHSMILES_OUTPUT_DIR:-outputs/runs/$RUN_NAME}"
TRAIN_FRACTION="${SKETCHSMILES_TRAIN_FRACTION:-0.8}"
SEED="${SKETCHSMILES_SEED:-7}"
LIMIT="${SKETCHSMILES_LIMIT:-}"
FINGERPRINT_BITS="${SKETCHSMILES_FINGERPRINT_BITS:-2048}"
MAX_LENGTH="${SKETCHSMILES_MAX_LENGTH:-128}"
HIDDEN_DIM="${SKETCHSMILES_HIDDEN_DIM:-384}"
EMBEDDING_DIM="${SKETCHSMILES_EMBEDDING_DIM:-96}"
EPOCHS="${SKETCHSMILES_EPOCHS:-20}"
BATCH_SIZE="${SKETCHSMILES_BATCH_SIZE:-128}"
LEARNING_RATE="${SKETCHSMILES_LEARNING_RATE:-0.001}"
SAMPLES_PER_CONDITION="${SKETCHSMILES_SAMPLES_PER_CONDITION:-8}"
TEMPERATURE="${SKETCHSMILES_TEMPERATURE:-0.9}"
SAMPLE_TOP_K="${SKETCHSMILES_SAMPLE_TOP_K:-16}"
IMAGE_SIZE="${SKETCHSMILES_IMAGE_SIZE:-256}"
SAMPLE_COUNT="${SKETCHSMILES_SAMPLE_COUNT:-64}"
CONTACT_SHEET_COLS="${SKETCHSMILES_CONTACT_SHEET_COLS:-8}"
CONTACT_THUMB_SIZE="${SKETCHSMILES_CONTACT_THUMB_SIZE:-144}"
DEVICE="${SKETCHSMILES_DEVICE:-auto}"

if [[ -n "${SKETCHSMILES_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHSMILES_MODULES
fi

echo "SketchSMILES Phase 5A-1 learned SMILES decoder"
echo "  python=$PYTHON_BIN"
echo "  modules=${SKETCHSMILES_MODULES:-<none>}"
echo "  pair_dir=$PAIR_DIR"
echo "  run_root=$OUTPUT_DIR"
echo "  train_fraction=$TRAIN_FRACTION"
echo "  seed=$SEED"
echo "  limit=${LIMIT:-<all>}"
echo "  fingerprint_bits=$FINGERPRINT_BITS"
echo "  max_length=$MAX_LENGTH"
echo "  hidden_dim=$HIDDEN_DIM"
echo "  embedding_dim=$EMBEDDING_DIM"
echo "  epochs=$EPOCHS"
echo "  batch_size=$BATCH_SIZE"
echo "  samples_per_condition=$SAMPLES_PER_CONDITION"
echo "  device=$DEVICE"

if [[ ! -f "$PAIR_DIR/pairs.csv" ]]; then
  echo "ERROR: pairs.csv not found under $PAIR_DIR" >&2
  echo "Run scripts/run_phase0_pairs.sh first, or set SKETCHSMILES_PAIR_DIR." >&2
  exit 2
fi

if [[ "${SKETCHSMILES_RUN_TESTS:-1}" == "1" ]]; then
  echo "[1/2] Running tests"
  "$PYTHON_BIN" -m unittest discover -s tests
  echo
else
  echo "[1/2] Skipping tests because SKETCHSMILES_RUN_TESTS=$SKETCHSMILES_RUN_TESTS"
  echo
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
import rdkit  # noqa: F401
import torch  # noqa: F401
from PIL import Image  # noqa: F401
PY
then
  echo "ERROR: Phase 5A-1 requires NumPy, RDKit, PyTorch, and Pillow." >&2
  echo "Try loading the server modules and using SKETCHSMILES_PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python." >&2
  exit 2
fi

ARGS=(
  -m sketch_smiles.phase5a1_learned_smiles_decoder
  --pair-dir "$PAIR_DIR"
  --output-dir "$OUTPUT_DIR"
  --train-fraction "$TRAIN_FRACTION"
  --seed "$SEED"
  --fingerprint-bits "$FINGERPRINT_BITS"
  --max-length "$MAX_LENGTH"
  --hidden-dim "$HIDDEN_DIM"
  --embedding-dim "$EMBEDDING_DIM"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --samples-per-condition "$SAMPLES_PER_CONDITION"
  --temperature "$TEMPERATURE"
  --sample-top-k "$SAMPLE_TOP_K"
  --image-size "$IMAGE_SIZE"
  --sample-count "$SAMPLE_COUNT"
  --contact-sheet-cols "$CONTACT_SHEET_COLS"
  --contact-thumb-size "$CONTACT_THUMB_SIZE"
  --device "$DEVICE"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

echo "[2/2] Training learned SMILES decoder and rendering top predictions"
"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "Phase 5A-1 learned SMILES decoder finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  predictions=$OUTPUT_DIR/predictions.csv"
echo "  model=$OUTPUT_DIR/model.pt"
echo "  vocab=$OUTPUT_DIR/vocab.json"
echo "  train_history=$OUTPUT_DIR/train_history.json"
echo "  sample_contact_sheet=$OUTPUT_DIR/sample_contact_sheet.png"
