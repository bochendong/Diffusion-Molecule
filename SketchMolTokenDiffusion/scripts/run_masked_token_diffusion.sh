#!/usr/bin/env bash
# Run Route A: masked token diffusion that emits SMILES directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SKETCHMOL_TOKEN_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PAIR_DIR="${SKETCHMOL_TOKEN_PAIR_DIR:-SketchSMILES/outputs/pairs/phys_50k}"
RUN_NAME="${SKETCHMOL_TOKEN_RUN_NAME:-token_diffusion_seed${SKETCHMOL_TOKEN_SEED:-7}}"
OUTPUT_DIR="${SKETCHMOL_TOKEN_OUTPUT_DIR:-SketchMolTokenDiffusion/outputs/runs/$RUN_NAME}"
TRAIN_FRACTION="${SKETCHMOL_TOKEN_TRAIN_FRACTION:-0.8}"
SEED="${SKETCHMOL_TOKEN_SEED:-7}"
LIMIT="${SKETCHMOL_TOKEN_LIMIT:-}"
FINGERPRINT_BITS="${SKETCHMOL_TOKEN_FINGERPRINT_BITS:-2048}"
MAX_LENGTH="${SKETCHMOL_TOKEN_MAX_LENGTH:-128}"
HIDDEN_DIM="${SKETCHMOL_TOKEN_HIDDEN_DIM:-384}"
EMBEDDING_DIM="${SKETCHMOL_TOKEN_EMBEDDING_DIM:-96}"
EPOCHS="${SKETCHMOL_TOKEN_EPOCHS:-20}"
BATCH_SIZE="${SKETCHMOL_TOKEN_BATCH_SIZE:-128}"
LEARNING_RATE="${SKETCHMOL_TOKEN_LEARNING_RATE:-0.001}"
DIFFUSION_STEPS="${SKETCHMOL_TOKEN_DIFFUSION_STEPS:-16}"
MIN_MASK_PROB="${SKETCHMOL_TOKEN_MIN_MASK_PROB:-0.15}"
MAX_MASK_PROB="${SKETCHMOL_TOKEN_MAX_MASK_PROB:-0.95}"
SAMPLES_PER_CONDITION="${SKETCHMOL_TOKEN_SAMPLES_PER_CONDITION:-8}"
TEMPERATURE="${SKETCHMOL_TOKEN_TEMPERATURE:-0.9}"
SAMPLE_TOP_K="${SKETCHMOL_TOKEN_SAMPLE_TOP_K:-16}"
RERANK_MODE="${SKETCHMOL_TOKEN_RERANK_MODE:-condition_fingerprint}"
TRANSFORMER_LAYERS="${SKETCHMOL_TOKEN_TRANSFORMER_LAYERS:-4}"
ATTENTION_HEADS="${SKETCHMOL_TOKEN_ATTENTION_HEADS:-8}"
CONDITION_TOKENS="${SKETCHMOL_TOKEN_CONDITION_TOKENS:-8}"
DROPOUT="${SKETCHMOL_TOKEN_DROPOUT:-0.1}"
TOKENIZATION="${SKETCHMOL_TOKEN_TOKENIZATION:-smiles_token}"
IMAGE_SIZE="${SKETCHMOL_TOKEN_IMAGE_SIZE:-128}"
SAMPLE_COUNT="${SKETCHMOL_TOKEN_SAMPLE_COUNT:-64}"
DEVICE="${SKETCHMOL_TOKEN_DEVICE:-auto}"

if [[ -n "${SKETCHMOL_TOKEN_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_TOKEN_MODULES
fi

export PYTHONPATH="$PROJECT_DIR:$REPO_ROOT/SketchSMILES${PYTHONPATH:+:$PYTHONPATH}"

echo "SketchMolTokenDiffusion Route A"
echo "  python=$PYTHON_BIN"
echo "  modules=${SKETCHMOL_TOKEN_MODULES:-<none>}"
echo "  pair_dir=$PAIR_DIR"
echo "  run_root=$OUTPUT_DIR"
echo "  epochs=$EPOCHS"
echo "  batch_size=$BATCH_SIZE"
echo "  diffusion_steps=$DIFFUSION_STEPS"
echo "  samples_per_condition=$SAMPLES_PER_CONDITION"
echo "  device=$DEVICE"

if [[ ! -f "$PAIR_DIR/pairs.csv" ]]; then
  echo "ERROR: pairs.csv not found under $PAIR_DIR" >&2
  exit 2
fi

if [[ "${SKETCHMOL_TOKEN_RUN_TESTS:-1}" == "1" ]]; then
  echo "[1/2] Running tests"
  "$PYTHON_BIN" -m unittest discover -s SketchMolTokenDiffusion/tests -p 'test_*.py'
  echo
else
  echo "[1/2] Skipping tests because SKETCHMOL_TOKEN_RUN_TESTS=$SKETCHMOL_TOKEN_RUN_TESTS"
  echo
fi

ARGS=(
  -m sketchmol_token_diffusion.masked_token_diffusion
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
  --diffusion-steps "$DIFFUSION_STEPS"
  --min-mask-prob "$MIN_MASK_PROB"
  --max-mask-prob "$MAX_MASK_PROB"
  --samples-per-condition "$SAMPLES_PER_CONDITION"
  --temperature "$TEMPERATURE"
  --sample-top-k "$SAMPLE_TOP_K"
  --rerank-mode "$RERANK_MODE"
  --transformer-layers "$TRANSFORMER_LAYERS"
  --attention-heads "$ATTENTION_HEADS"
  --condition-tokens "$CONDITION_TOKENS"
  --dropout "$DROPOUT"
  --tokenization "$TOKENIZATION"
  --image-size "$IMAGE_SIZE"
  --sample-count "$SAMPLE_COUNT"
  --device "$DEVICE"
  --route-name "sketchmol_token_diffusion_masked_smiles"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

echo "[2/2] Training masked token diffusion and rendering top predictions"
"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "SketchMolTokenDiffusion finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  predictions=$OUTPUT_DIR/predictions.csv"
