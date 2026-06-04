#!/usr/bin/env bash
# Train on the original SketchMol before/after optimization examples.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${LETA_PYTHON_BIN:-${PYTHON_BIN:-/home/bdong/scratch/venvs/phystabmol/bin/python}}"
RUN_NAME="${LETA_RUN_NAME:-sketchmol_opt_pairs_seed${LETA_SEED:-7}}"
OUTPUT_DIR="${LETA_OUTPUT_DIR:-outputs/runs/$RUN_NAME}"
MODEL_KIND="${LETA_MODEL_KIND:-history}"
OPT_EXAMPLES_DIR="${LETA_OPT_EXAMPLES_DIR:-/home/bdong/scratch/projects/Diffusion-Molecule/Research/Molecule Generation/SketchMol/SketchMol-v1-main/opt_examples}"
SEED="${LETA_SEED:-7}"
MAX_EXAMPLES="${LETA_MAX_EXAMPLES:-}"
LATENT_DIM="${LETA_LATENT_DIM:-256}"
PROPERTY_DIM="${LETA_PROPERTY_DIM:-4}"
TARGET_DIM="${LETA_TARGET_DIM:-4}"
EDIT_TYPE_COUNT="${LETA_EDIT_TYPE_COUNT:-16}"
HIDDEN_DIM="${LETA_HIDDEN_DIM:-256}"
TRANSFORMER_LAYERS="${LETA_TRANSFORMER_LAYERS:-4}"
ATTENTION_HEADS="${LETA_ATTENTION_HEADS:-8}"
DIFFUSION_STEPS="${LETA_DIFFUSION_STEPS:-100}"
MAX_HISTORY="${LETA_MAX_HISTORY:-4}"
DROPOUT="${LETA_DROPOUT:-0.1}"
EPOCHS="${LETA_EPOCHS:-30}"
BATCH_SIZE="${LETA_BATCH_SIZE:-64}"
LEARNING_RATE="${LETA_LEARNING_RATE:-0.001}"
DEVICE="${LETA_DEVICE:-auto}"
FINGERPRINT_RADIUS="${LETA_FINGERPRINT_RADIUS:-2}"
MODULES="${LETA_MODULES:-gcc rdkit/2025.09.4}"

if [[ -n "$MODULES" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Latent Edit Trajectory Attention SketchMol opt-pair training"
echo "  python=$PYTHON_BIN"
echo "  modules=${MODULES:-<none>}"
echo "  opt_examples_dir=$OPT_EXAMPLES_DIR"
echo "  output_dir=$OUTPUT_DIR"
echo "  model_kind=$MODEL_KIND"
echo "  latent_dim=$LATENT_DIM"
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

ARGS=(
  -m latent_edit_trajectory_attention.train
  --dataset sketchmol_opt
  --model-kind "$MODEL_KIND"
  --output-dir "$OUTPUT_DIR"
  --opt-examples-dir "$OPT_EXAMPLES_DIR"
  --seed "$SEED"
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

echo "[2/2] Training on SketchMol opt_examples"
"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "SketchMol opt-pair training finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  history=$OUTPUT_DIR/train_history.json"

