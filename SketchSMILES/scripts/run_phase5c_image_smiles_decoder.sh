#!/usr/bin/env bash
# Run Phase 5C image-conditioned SMILES decoder.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHSMILES_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PAIR_DIR="${SKETCHSMILES_PAIR_DIR:-outputs/pairs/phys_50k}"
RUN_NAME="${SKETCHSMILES_RUN_NAME:-phase5c_image_smiles_decoder_seed${SKETCHSMILES_SEED:-7}}"
OUTPUT_DIR="${SKETCHSMILES_OUTPUT_DIR:-outputs/runs/$RUN_NAME}"
TRAIN_FRACTION="${SKETCHSMILES_TRAIN_FRACTION:-0.8}"
SEED="${SKETCHSMILES_SEED:-7}"
LIMIT="${SKETCHSMILES_LIMIT:-}"
MAX_LENGTH="${SKETCHSMILES_MAX_LENGTH:-128}"
HIDDEN_DIM="${SKETCHSMILES_HIDDEN_DIM:-384}"
EMBEDDING_DIM="${SKETCHSMILES_EMBEDDING_DIM:-96}"
ENCODER_CHANNELS="${SKETCHSMILES_ENCODER_CHANNELS:-64}"
IMAGE_TOKEN_GRID="${SKETCHSMILES_IMAGE_TOKEN_GRID:-4}"
TRANSFORMER_LAYERS="${SKETCHSMILES_TRANSFORMER_LAYERS:-4}"
ATTENTION_HEADS="${SKETCHSMILES_ATTENTION_HEADS:-8}"
DROPOUT="${SKETCHSMILES_DROPOUT:-0.1}"
EPOCHS="${SKETCHSMILES_EPOCHS:-20}"
BATCH_SIZE="${SKETCHSMILES_BATCH_SIZE:-128}"
LEARNING_RATE="${SKETCHSMILES_LEARNING_RATE:-0.001}"
SAMPLES_PER_CONDITION="${SKETCHSMILES_SAMPLES_PER_CONDITION:-8}"
TEMPERATURE="${SKETCHSMILES_TEMPERATURE:-0.9}"
SAMPLE_TOP_K="${SKETCHSMILES_SAMPLE_TOP_K:-16}"
TOKENIZATION="${SKETCHSMILES_TOKENIZATION:-smiles_token}"
DECODING="${SKETCHSMILES_DECODING:-beam}"
BEAM_SIZE="${SKETCHSMILES_BEAM_SIZE:-8}"
LENGTH_PENALTY="${SKETCHSMILES_LENGTH_PENALTY:-0.0}"
IMAGE_SIZE="${SKETCHSMILES_IMAGE_SIZE:-128}"
SAMPLE_COUNT="${SKETCHSMILES_SAMPLE_COUNT:-64}"
CONTACT_SHEET_COLS="${SKETCHSMILES_CONTACT_SHEET_COLS:-8}"
CONTACT_THUMB_SIZE="${SKETCHSMILES_CONTACT_THUMB_SIZE:-144}"
DEVICE="${SKETCHSMILES_DEVICE:-auto}"

if [[ -n "${SKETCHSMILES_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHSMILES_MODULES
fi

echo "SketchSMILES Phase 5C image-conditioned SMILES decoder"
echo "  python=$PYTHON_BIN"
echo "  modules=${SKETCHSMILES_MODULES:-<none>}"
echo "  pair_dir=$PAIR_DIR"
echo "  run_root=$OUTPUT_DIR"
echo "  train_fraction=$TRAIN_FRACTION"
echo "  seed=$SEED"
echo "  limit=${LIMIT:-<all>}"
echo "  max_length=$MAX_LENGTH"
echo "  hidden_dim=$HIDDEN_DIM"
echo "  embedding_dim=$EMBEDDING_DIM"
echo "  encoder_channels=$ENCODER_CHANNELS"
echo "  image_token_grid=$IMAGE_TOKEN_GRID"
echo "  transformer_layers=$TRANSFORMER_LAYERS"
echo "  attention_heads=$ATTENTION_HEADS"
echo "  dropout=$DROPOUT"
echo "  epochs=$EPOCHS"
echo "  batch_size=$BATCH_SIZE"
echo "  tokenization=$TOKENIZATION"
echo "  decoding=$DECODING"
echo "  beam_size=$BEAM_SIZE"
echo "  image_size=$IMAGE_SIZE"
echo "  device=$DEVICE"

"$PYTHON_BIN" - <<'PY'
import torch

print(f"  torch={torch.__version__}")
print(f"  torch_cuda_available={torch.cuda.is_available()}")
print(f"  torch_cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

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
  echo "ERROR: Phase 5C requires NumPy, RDKit, PyTorch, and Pillow." >&2
  exit 2
fi

ARGS=(
  -m sketch_smiles.phase5c_image_smiles_decoder
  --pair-dir "$PAIR_DIR"
  --output-dir "$OUTPUT_DIR"
  --train-fraction "$TRAIN_FRACTION"
  --seed "$SEED"
  --max-length "$MAX_LENGTH"
  --hidden-dim "$HIDDEN_DIM"
  --embedding-dim "$EMBEDDING_DIM"
  --encoder-channels "$ENCODER_CHANNELS"
  --image-token-grid "$IMAGE_TOKEN_GRID"
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

echo "[2/2] Training image-conditioned SMILES decoder and rendering top predictions"
"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "Phase 5C image-conditioned SMILES decoder finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  predictions=$OUTPUT_DIR/predictions.csv"
echo "  model=$OUTPUT_DIR/model.pt"
echo "  vocab=$OUTPUT_DIR/vocab.json"
echo "  train_history=$OUTPUT_DIR/train_history.json"
echo "  sample_contact_sheet=$OUTPUT_DIR/sample_contact_sheet.png"
