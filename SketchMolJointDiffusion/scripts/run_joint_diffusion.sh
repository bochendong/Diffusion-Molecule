#!/usr/bin/env bash
# Run Route B: masked token diffusion with a learned image head.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SKETCHMOL_JOINT_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
PAIR_DIR="${SKETCHMOL_JOINT_PAIR_DIR:-SketchSMILES/outputs/pairs/phys_50k}"
RUN_NAME="${SKETCHMOL_JOINT_RUN_NAME:-joint_diffusion_seed${SKETCHMOL_JOINT_SEED:-7}}"
OUTPUT_DIR="${SKETCHMOL_JOINT_OUTPUT_DIR:-SketchMolJointDiffusion/outputs/runs/$RUN_NAME}"
TRAIN_FRACTION="${SKETCHMOL_JOINT_TRAIN_FRACTION:-0.8}"
SEED="${SKETCHMOL_JOINT_SEED:-7}"
LIMIT="${SKETCHMOL_JOINT_LIMIT:-}"
FINGERPRINT_BITS="${SKETCHMOL_JOINT_FINGERPRINT_BITS:-2048}"
MAX_LENGTH="${SKETCHMOL_JOINT_MAX_LENGTH:-128}"
HIDDEN_DIM="${SKETCHMOL_JOINT_HIDDEN_DIM:-384}"
EMBEDDING_DIM="${SKETCHMOL_JOINT_EMBEDDING_DIM:-96}"
EPOCHS="${SKETCHMOL_JOINT_EPOCHS:-20}"
BATCH_SIZE="${SKETCHMOL_JOINT_BATCH_SIZE:-128}"
LEARNING_RATE="${SKETCHMOL_JOINT_LEARNING_RATE:-0.001}"
DIFFUSION_STEPS="${SKETCHMOL_JOINT_DIFFUSION_STEPS:-16}"
MIN_MASK_PROB="${SKETCHMOL_JOINT_MIN_MASK_PROB:-0.15}"
MAX_MASK_PROB="${SKETCHMOL_JOINT_MAX_MASK_PROB:-0.95}"
SAMPLES_PER_CONDITION="${SKETCHMOL_JOINT_SAMPLES_PER_CONDITION:-8}"
TEMPERATURE="${SKETCHMOL_JOINT_TEMPERATURE:-0.9}"
SAMPLE_TOP_K="${SKETCHMOL_JOINT_SAMPLE_TOP_K:-16}"
RERANK_MODE="${SKETCHMOL_JOINT_RERANK_MODE:-condition_fingerprint}"
TRANSFORMER_LAYERS="${SKETCHMOL_JOINT_TRANSFORMER_LAYERS:-4}"
ATTENTION_HEADS="${SKETCHMOL_JOINT_ATTENTION_HEADS:-8}"
CONDITION_TOKENS="${SKETCHMOL_JOINT_CONDITION_TOKENS:-8}"
DROPOUT="${SKETCHMOL_JOINT_DROPOUT:-0.1}"
TOKENIZATION="${SKETCHMOL_JOINT_TOKENIZATION:-smiles_token}"
LATENT_DIM="${SKETCHMOL_JOINT_LATENT_DIM:-128}"
IMAGE_LOSS_WEIGHT="${SKETCHMOL_JOINT_IMAGE_LOSS_WEIGHT:-1.0}"
IMAGE_FOREGROUND_WEIGHT="${SKETCHMOL_JOINT_IMAGE_FOREGROUND_WEIGHT:-8.0}"
CLIP_LOSS_WEIGHT="${SKETCHMOL_JOINT_CLIP_LOSS_WEIGHT:-0.0}"
CLIP_TEMPERATURE="${SKETCHMOL_JOINT_CLIP_TEMPERATURE:-0.07}"
DECODE_LENGTH_MODE="${SKETCHMOL_JOINT_DECODE_LENGTH_MODE:-free}"
MIN_DECODE_TOKENS="${SKETCHMOL_JOINT_MIN_DECODE_TOKENS:-1}"
IMAGE_SIZE="${SKETCHMOL_JOINT_IMAGE_SIZE:-128}"
SAMPLE_COUNT="${SKETCHMOL_JOINT_SAMPLE_COUNT:-64}"
DEVICE="${SKETCHMOL_JOINT_DEVICE:-auto}"

if [[ -n "${SKETCHMOL_JOINT_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_JOINT_MODULES
fi

export PYTHONPATH="$PROJECT_DIR:$REPO_ROOT/SketchMolTokenDiffusion:$REPO_ROOT/SketchSMILES${PYTHONPATH:+:$PYTHONPATH}"

echo "SketchMolJointDiffusion Route B"
echo "  python=$PYTHON_BIN"
echo "  modules=${SKETCHMOL_JOINT_MODULES:-<none>}"
echo "  pair_dir=$PAIR_DIR"
echo "  run_root=$OUTPUT_DIR"
echo "  epochs=$EPOCHS"
echo "  batch_size=$BATCH_SIZE"
echo "  diffusion_steps=$DIFFUSION_STEPS"
echo "  tokenization=$TOKENIZATION"
echo "  image_loss_weight=$IMAGE_LOSS_WEIGHT"
echo "  clip_loss_weight=$CLIP_LOSS_WEIGHT"
echo "  decode_length_mode=$DECODE_LENGTH_MODE"
echo "  device=$DEVICE"

if [[ ! -f "$PAIR_DIR/pairs.csv" ]]; then
  echo "ERROR: pairs.csv not found under $PAIR_DIR" >&2
  exit 2
fi

if [[ "$TOKENIZATION" == "selfies" ]]; then
  if ! "$PYTHON_BIN" -c "import selfies" >/dev/null 2>&1; then
    echo "ERROR: SELFIES tokenization requires the selfies package." >&2
    echo "Install it with: $PYTHON_BIN -m pip install selfies" >&2
    exit 2
  fi
fi

if [[ "${SKETCHMOL_JOINT_RUN_TESTS:-1}" == "1" ]]; then
  echo "[1/2] Running tests"
  "$PYTHON_BIN" -m unittest discover -s SketchMolJointDiffusion/tests -p 'test_*.py'
  "$PYTHON_BIN" -m unittest discover -s SketchMolTokenDiffusion/tests -p 'test_*.py'
  echo
else
  echo "[1/2] Skipping tests because SKETCHMOL_JOINT_RUN_TESTS=$SKETCHMOL_JOINT_RUN_TESTS"
  echo
fi

ARGS=(
  -m sketchmol_joint_diffusion.joint_diffusion
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
  --latent-dim "$LATENT_DIM"
  --image-loss-weight "$IMAGE_LOSS_WEIGHT"
  --image-foreground-weight "$IMAGE_FOREGROUND_WEIGHT"
  --clip-loss-weight "$CLIP_LOSS_WEIGHT"
  --clip-temperature "$CLIP_TEMPERATURE"
  --decode-length-mode "$DECODE_LENGTH_MODE"
  --min-decode-tokens "$MIN_DECODE_TOKENS"
  --image-size "$IMAGE_SIZE"
  --sample-count "$SAMPLE_COUNT"
  --device "$DEVICE"
  --route-name "sketchmol_joint_diffusion_image_smiles"
)
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

echo "[2/2] Training joint image+SMILES diffusion and evaluating consistency"
"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "SketchMolJointDiffusion finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  predictions=$OUTPUT_DIR/predictions.csv"
