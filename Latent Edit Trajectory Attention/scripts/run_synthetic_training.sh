#!/usr/bin/env bash
# Run the synthetic Latent Edit Trajectory Attention training job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${LETA_PYTHON_BIN:-${PYTHON_BIN:-/home/bdong/scratch/venvs/phystabmol/bin/python}}"
RUN_NAME="${LETA_RUN_NAME:-trajectory_attention_seed${LETA_SEED:-7}}"
OUTPUT_DIR="${LETA_OUTPUT_DIR:-outputs/runs/$RUN_NAME}"
SEED="${LETA_SEED:-7}"
EXAMPLES="${LETA_EXAMPLES:-4096}"
HISTORY_LENGTH="${LETA_HISTORY_LENGTH:-8}"
LATENT_DIM="${LETA_LATENT_DIM:-128}"
PROPERTY_DIM="${LETA_PROPERTY_DIM:-4}"
TARGET_DIM="${LETA_TARGET_DIM:-4}"
EDIT_TYPE_COUNT="${LETA_EDIT_TYPE_COUNT:-16}"
HIDDEN_DIM="${LETA_HIDDEN_DIM:-256}"
TRANSFORMER_LAYERS="${LETA_TRANSFORMER_LAYERS:-4}"
ATTENTION_HEADS="${LETA_ATTENTION_HEADS:-8}"
DIFFUSION_STEPS="${LETA_DIFFUSION_STEPS:-100}"
MAX_HISTORY="${LETA_MAX_HISTORY:-16}"
DROPOUT="${LETA_DROPOUT:-0.1}"
EPOCHS="${LETA_EPOCHS:-20}"
BATCH_SIZE="${LETA_BATCH_SIZE:-128}"
LEARNING_RATE="${LETA_LEARNING_RATE:-0.001}"
DEVICE="${LETA_DEVICE:-auto}"

if [[ -n "${LETA_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $LETA_MODULES
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Latent Edit Trajectory Attention synthetic training"
echo "  python=$PYTHON_BIN"
echo "  modules=${LETA_MODULES:-<none>}"
echo "  output_dir=$OUTPUT_DIR"
echo "  examples=$EXAMPLES"
echo "  history_length=$HISTORY_LENGTH"
echo "  latent_dim=$LATENT_DIM"
echo "  hidden_dim=$HIDDEN_DIM"
echo "  epochs=$EPOCHS"
echo "  batch_size=$BATCH_SIZE"
echo "  device=$DEVICE"

if [[ "${LETA_RUN_TESTS:-1}" == "1" ]]; then
  echo "[1/2] Running tests"
  "$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py'
  echo
else
  echo "[1/2] Skipping tests because LETA_RUN_TESTS=$LETA_RUN_TESTS"
  echo
fi

echo "[2/2] Training trajectory-conditioned latent diffusion editor"
"$PYTHON_BIN" -m latent_edit_trajectory_attention.train \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --examples "$EXAMPLES" \
  --history-length "$HISTORY_LENGTH" \
  --latent-dim "$LATENT_DIM" \
  --property-dim "$PROPERTY_DIM" \
  --target-dim "$TARGET_DIM" \
  --edit-type-count "$EDIT_TYPE_COUNT" \
  --hidden-dim "$HIDDEN_DIM" \
  --transformer-layers "$TRANSFORMER_LAYERS" \
  --attention-heads "$ATTENTION_HEADS" \
  --diffusion-steps "$DIFFUSION_STEPS" \
  --max-history "$MAX_HISTORY" \
  --dropout "$DROPOUT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --device "$DEVICE"

echo
echo "Latent Edit Trajectory Attention finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  history=$OUTPUT_DIR/train_history.json"

