#!/usr/bin/env bash
# Train and evaluate an image-molecule contrastive reranker.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SKETCHMOL_E2E_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PAIR_DIR="${SKETCHMOL_E2E_PAIR_DIR:-SketchSMILES/outputs/pairs/phys_50k}"
RUN_DIR="${SKETCHMOL_E2E_RUN_DIR:-SketchMolEnd2End/outputs/runs/image_to_structure_fp_aux_seed7}"
TRAIN_PAIRS_CSV="${SKETCHMOL_E2E_TRAIN_PAIRS_CSV:-$RUN_DIR/train_pairs.csv}"
PREDICTIONS_CSV="${SKETCHMOL_E2E_PREDICTIONS_CSV:-$RUN_DIR/predictions.csv}"
OUTPUT_DIR="${SKETCHMOL_E2E_CONTRASTIVE_OUTPUT_DIR:-$RUN_DIR/contrastive_reranker}"
FINGERPRINT_BITS="${SKETCHMOL_E2E_FINGERPRINT_BITS:-2048}"
IMAGE_SIZE="${SKETCHMOL_E2E_IMAGE_SIZE:-128}"
EMBEDDING_DIM="${SKETCHMOL_E2E_CONTRASTIVE_EMBEDDING_DIM:-256}"
ENCODER_CHANNELS="${SKETCHMOL_E2E_ENCODER_CHANNELS:-64}"
HIDDEN_DIM="${SKETCHMOL_E2E_HIDDEN_DIM:-512}"
EPOCHS="${SKETCHMOL_E2E_EPOCHS:-5}"
BATCH_SIZE="${SKETCHMOL_E2E_BATCH_SIZE:-128}"
LEARNING_RATE="${SKETCHMOL_E2E_LEARNING_RATE:-0.001}"
TRAIN_LIMIT="${SKETCHMOL_E2E_TRAIN_LIMIT:-}"
EVAL_LIMIT="${SKETCHMOL_E2E_EVAL_LIMIT:-}"
SEED="${SKETCHMOL_E2E_SEED:-7}"
DEVICE="${SKETCHMOL_E2E_DEVICE:-auto}"

export PYTHONPATH="$PROJECT_DIR:$REPO_ROOT/SketchSMILES${PYTHONPATH:+:$PYTHONPATH}"
export SKETCHMOL_E2E_MODULES="${SKETCHMOL_E2E_MODULES:-gcc rdkit/2025.09.4}"

if [[ -n "${SKETCHMOL_E2E_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_E2E_MODULES
fi

echo "SketchMolEnd2End contrastive reranker"
echo "  python=$PYTHON_BIN"
echo "  pair_dir=$PAIR_DIR"
echo "  train_pairs=$TRAIN_PAIRS_CSV"
echo "  predictions=$PREDICTIONS_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  epochs=$EPOCHS"
echo "  batch_size=$BATCH_SIZE"
echo "  image_size=$IMAGE_SIZE"
echo "  train_limit=${TRAIN_LIMIT:-<all>}"
echo "  eval_limit=${EVAL_LIMIT:-<all>}"
echo "  device=$DEVICE"

if [[ ! -f "$TRAIN_PAIRS_CSV" ]]; then
  echo "ERROR: train pairs not found: $TRAIN_PAIRS_CSV" >&2
  exit 2
fi
if [[ ! -f "$PREDICTIONS_CSV" ]]; then
  echo "ERROR: predictions not found: $PREDICTIONS_CSV" >&2
  exit 2
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
import rdkit  # noqa: F401
import torch  # noqa: F401
from PIL import Image  # noqa: F401
PY
then
  echo "ERROR: contrastive reranker requires NumPy, RDKit, PyTorch, and Pillow." >&2
  exit 2
fi

ARGS=(
  -m sketchmol_end2end.contrastive_reranker
  --train-pairs-csv "$TRAIN_PAIRS_CSV"
  --pair-dir "$PAIR_DIR"
  --predictions-csv "$PREDICTIONS_CSV"
  --output-dir "$OUTPUT_DIR"
  --fingerprint-bits "$FINGERPRINT_BITS"
  --image-size "$IMAGE_SIZE"
  --embedding-dim "$EMBEDDING_DIM"
  --encoder-channels "$ENCODER_CHANNELS"
  --hidden-dim "$HIDDEN_DIM"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --seed "$SEED"
  --device "$DEVICE"
)
if [[ -n "$TRAIN_LIMIT" ]]; then
  ARGS+=(--train-limit "$TRAIN_LIMIT")
fi
if [[ -n "$EVAL_LIMIT" ]]; then
  ARGS+=(--eval-limit "$EVAL_LIMIT")
fi

"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "Contrastive reranker finished: $OUTPUT_DIR"
echo "  summary=$OUTPUT_DIR/contrastive_rerank_summary.json"
echo "  predictions=$OUTPUT_DIR/contrastive_reranked_predictions.csv"
