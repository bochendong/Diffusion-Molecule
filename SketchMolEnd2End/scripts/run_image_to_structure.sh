#!/usr/bin/env bash
# Run the no-OCR end-to-end image-to-structure baseline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SKETCHMOL_E2E_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PAIR_DIR="${SKETCHMOL_E2E_PAIR_DIR:-SketchSMILES/outputs/pairs/phys_50k}"
SEED="${SKETCHMOL_E2E_SEED:-7}"
RUN_NAME="${SKETCHMOL_E2E_RUN_NAME:-image_to_structure_seed${SEED}}"
OUTPUT_DIR="${SKETCHMOL_E2E_OUTPUT_DIR:-SketchMolEnd2End/outputs/runs/$RUN_NAME}"
TRAIN_FRACTION="${SKETCHMOL_E2E_TRAIN_FRACTION:-0.8}"
LIMIT="${SKETCHMOL_E2E_LIMIT:-}"
MAX_LENGTH="${SKETCHMOL_E2E_MAX_LENGTH:-128}"
HIDDEN_DIM="${SKETCHMOL_E2E_HIDDEN_DIM:-384}"
EMBEDDING_DIM="${SKETCHMOL_E2E_EMBEDDING_DIM:-96}"
ENCODER_CHANNELS="${SKETCHMOL_E2E_ENCODER_CHANNELS:-64}"
IMAGE_TOKEN_GRID="${SKETCHMOL_E2E_IMAGE_TOKEN_GRID:-4}"
FINGERPRINT_BITS="${SKETCHMOL_E2E_FINGERPRINT_BITS:-0}"
FINGERPRINT_LOSS_WEIGHT="${SKETCHMOL_E2E_FINGERPRINT_LOSS_WEIGHT:-0.0}"
RERANK_MODE="${SKETCHMOL_E2E_RERANK_MODE:-beam}"
TRANSFORMER_LAYERS="${SKETCHMOL_E2E_TRANSFORMER_LAYERS:-4}"
ATTENTION_HEADS="${SKETCHMOL_E2E_ATTENTION_HEADS:-8}"
DROPOUT="${SKETCHMOL_E2E_DROPOUT:-0.1}"
EPOCHS="${SKETCHMOL_E2E_EPOCHS:-20}"
BATCH_SIZE="${SKETCHMOL_E2E_BATCH_SIZE:-128}"
LEARNING_RATE="${SKETCHMOL_E2E_LEARNING_RATE:-0.001}"
SAMPLES_PER_CONDITION="${SKETCHMOL_E2E_SAMPLES_PER_CONDITION:-8}"
TEMPERATURE="${SKETCHMOL_E2E_TEMPERATURE:-0.9}"
SAMPLE_TOP_K="${SKETCHMOL_E2E_SAMPLE_TOP_K:-16}"
TOKENIZATION="${SKETCHMOL_E2E_TOKENIZATION:-smiles_token}"
DECODING="${SKETCHMOL_E2E_DECODING:-beam}"
BEAM_SIZE="${SKETCHMOL_E2E_BEAM_SIZE:-8}"
LENGTH_PENALTY="${SKETCHMOL_E2E_LENGTH_PENALTY:-0.0}"
IMAGE_SIZE="${SKETCHMOL_E2E_IMAGE_SIZE:-128}"
SAMPLE_COUNT="${SKETCHMOL_E2E_SAMPLE_COUNT:-64}"
CONTACT_SHEET_COLS="${SKETCHMOL_E2E_CONTACT_SHEET_COLS:-8}"
CONTACT_THUMB_SIZE="${SKETCHMOL_E2E_CONTACT_THUMB_SIZE:-144}"
DEVICE="${SKETCHMOL_E2E_DEVICE:-auto}"

export PYTHONPATH="$PROJECT_DIR:$REPO_ROOT/SketchSMILES${PYTHONPATH:+:$PYTHONPATH}"

export SKETCHMOL_E2E_MODULES="${SKETCHMOL_E2E_MODULES:-gcc rdkit/2025.09.4}"

if [[ -n "${SKETCHMOL_E2E_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_E2E_MODULES
fi

echo "SketchMolEnd2End image-to-structure baseline"
echo "  python=$PYTHON_BIN"
echo "  modules=${SKETCHMOL_E2E_MODULES:-<none>}"
echo "  pair_dir=$PAIR_DIR"
echo "  output_dir=$OUTPUT_DIR"
echo "  train_fraction=$TRAIN_FRACTION"
echo "  seed=$SEED"
echo "  limit=${LIMIT:-<all>}"
echo "  tokenization=$TOKENIZATION"
echo "  decoding=$DECODING"
echo "  epochs=$EPOCHS"
echo "  batch_size=$BATCH_SIZE"
echo "  image_size=$IMAGE_SIZE"
echo "  device=$DEVICE"

if [[ ! -f "$PAIR_DIR/pairs.csv" ]]; then
  echo "ERROR: pairs.csv not found under $PAIR_DIR" >&2
  echo "Run the SketchSMILES pair generation first, or set SKETCHMOL_E2E_PAIR_DIR." >&2
  exit 2
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
import rdkit  # noqa: F401
import torch  # noqa: F401
from PIL import Image  # noqa: F401
PY
then
  echo "ERROR: SketchMolEnd2End requires NumPy, RDKit, PyTorch, and Pillow." >&2
  exit 2
fi

ARGS=(
  -m sketchmol_end2end.image_to_structure
  --pair-dir "$PAIR_DIR"
  --output-dir "$OUTPUT_DIR"
  --train-fraction "$TRAIN_FRACTION"
  --seed "$SEED"
  --max-length "$MAX_LENGTH"
  --hidden-dim "$HIDDEN_DIM"
  --embedding-dim "$EMBEDDING_DIM"
  --encoder-channels "$ENCODER_CHANNELS"
  --image-token-grid "$IMAGE_TOKEN_GRID"
  --fingerprint-bits "$FINGERPRINT_BITS"
  --fingerprint-loss-weight "$FINGERPRINT_LOSS_WEIGHT"
  --rerank-mode "$RERANK_MODE"
  --transformer-layers "$TRANSFORMER_LAYERS"
  --attention-heads "$ATTENTION_HEADS"
  --dropout "$DROPOUT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --samples-per-condition "$SAMPLES_PER_CONDITION"
  --temperature "$TEMPERATURE"
  --sample-top-k "$SAMPLE_TOP_K"
  --tokenization "$TOKENIZATION"
  --decoding "$DECODING"
  --beam-size "$BEAM_SIZE"
  --length-penalty "$LENGTH_PENALTY"
  --image-size "$IMAGE_SIZE"
  --sample-count "$SAMPLE_COUNT"
  --contact-sheet-cols "$CONTACT_SHEET_COLS"
  --contact-thumb-size "$CONTACT_THUMB_SIZE"
  --device "$DEVICE"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

echo "[1/1] Training no-OCR image-to-structure model and rendering predictions"
"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "SketchMolEnd2End finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  predictions=$OUTPUT_DIR/predictions.csv"
echo "  manifest=$OUTPUT_DIR/end2end_manifest.json"
