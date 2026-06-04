#!/usr/bin/env bash
# Train one model variant on a trajectory JSONL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${LETA_PYTHON_BIN:-${PYTHON_BIN:-/home/bdong/scratch/venvs/phystabmol/bin/python}}"
MODULES="${LETA_MODULES:-gcc rdkit/2025.09.4}"
TRAJECTORY_PATH="${LETA_TRAJECTORY_PATH:-outputs/trajectories/sketchmol_opt_bootstrap.jsonl}"
MODEL_KIND="${LETA_MODEL_KIND:-history}"
RUN_NAME="${LETA_RUN_NAME:-sketchmol_trajectory_${MODEL_KIND}_seed${LETA_SEED:-7}}"
OUTPUT_DIR="${LETA_OUTPUT_DIR:-outputs/runs/$RUN_NAME}"
SEED="${LETA_SEED:-7}"
MAX_EXAMPLES="${LETA_MAX_EXAMPLES:-}"
MIN_HISTORY="${LETA_MIN_HISTORY:-1}"
LATENT_DIM="${LETA_LATENT_DIM:-256}"
PROPERTY_DIM="${LETA_PROPERTY_DIM:-5}"
TARGET_DIM="${LETA_TARGET_DIM:-4}"
EDIT_TYPE_COUNT="${LETA_EDIT_TYPE_COUNT:-16}"
HIDDEN_DIM="${LETA_HIDDEN_DIM:-256}"
TRANSFORMER_LAYERS="${LETA_TRANSFORMER_LAYERS:-4}"
ATTENTION_HEADS="${LETA_ATTENTION_HEADS:-8}"
DIFFUSION_STEPS="${LETA_DIFFUSION_STEPS:-100}"
MAX_HISTORY="${LETA_MAX_HISTORY:-8}"
DROPOUT="${LETA_DROPOUT:-0.1}"
EPOCHS="${LETA_EPOCHS:-20}"
BATCH_SIZE="${LETA_BATCH_SIZE:-64}"
LEARNING_RATE="${LETA_LEARNING_RATE:-0.001}"
DEVICE="${LETA_DEVICE:-cpu}"
FINGERPRINT_RADIUS="${LETA_FINGERPRINT_RADIUS:-2}"

if [[ -n "$MODULES" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  -m latent_edit_trajectory_attention.train
  --dataset sketchmol_trajectory
  --model-kind "$MODEL_KIND"
  --output-dir "$OUTPUT_DIR"
  --trajectory-path "$TRAJECTORY_PATH"
  --seed "$SEED"
  --min-history "$MIN_HISTORY"
  --fingerprint-radius "$FINGERPRINT_RADIUS"
  --latent-dim "$LATENT_DIM"
  --property-dim "$PROPERTY_DIM"
  --target-dim "$TARGET_DIM"
  --edit-type-count "$EDIT_TYPE_COUNT"
  --hidden-dim "$HIDDEN_DIM"
  --transformer-layers "$TRANSFORMER_LAYERS"
  --attention-heads "$ATTENTION_HEADS"
  --diffusion-steps "$DIFFUSION_STEPS"
  --max-history "$MAX_HISTORY"
  --dropout "$DROPOUT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --device "$DEVICE"
)
if [[ -n "$MAX_EXAMPLES" ]]; then
  ARGS+=(--max-examples "$MAX_EXAMPLES")
fi

echo "Latent Edit Trajectory Attention trajectory training"
echo "  python=$PYTHON_BIN"
echo "  trajectory_path=$TRAJECTORY_PATH"
echo "  model_kind=$MODEL_KIND"
echo "  output_dir=$OUTPUT_DIR"
echo "  epochs=$EPOCHS"
echo "  device=$DEVICE"
"$PYTHON_BIN" "${ARGS[@]}"

