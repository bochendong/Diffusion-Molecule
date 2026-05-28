#!/usr/bin/env bash
# Run Phase 2C: calibrate planner latents before decoder sampling.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
RUN_NAME="${SKETCHIMAGE_RUN_NAME:-phase2_calibrated_decoder_2048_seed${SKETCHIMAGE_SEED:-7}}"
OUTPUT_DIR="${SKETCHIMAGE_OUTPUT_DIR:-outputs/runs/$RUN_NAME}"

export SKETCHIMAGE_SEED="${SKETCHIMAGE_SEED:-7}"
export SKETCHIMAGE_DECODER_DIR="${SKETCHIMAGE_DECODER_DIR:-outputs/runs/phase2_robust_decoder_2048_seed${SKETCHIMAGE_SEED}/decoder}"
export SKETCHIMAGE_DECODER_POOL_DIR="${SKETCHIMAGE_DECODER_POOL_DIR:-outputs/runs/phase1_oracle_latent_ar_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_TRAIN_CSV="${SKETCHIMAGE_TRAIN_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED}_train.csv}"
export SKETCHIMAGE_EVAL_CSV="${SKETCHIMAGE_EVAL_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED}_eval.csv}"
export SKETCHIMAGE_FEATURE_DIM="${SKETCHIMAGE_FEATURE_DIM:-256}"
export SKETCHIMAGE_TOP_K="${SKETCHIMAGE_TOP_K:-8}"
export SKETCHIMAGE_RIDGE="${SKETCHIMAGE_RIDGE:-0.001}"
export SKETCHIMAGE_BACKEND="${SKETCHIMAGE_BACKEND:-torch_denoiser}"
export SKETCHIMAGE_TORCH_HIDDEN_DIM="${SKETCHIMAGE_TORCH_HIDDEN_DIM:-1024}"
export SKETCHIMAGE_TORCH_EPOCHS="${SKETCHIMAGE_TORCH_EPOCHS:-25}"
export SKETCHIMAGE_TORCH_BATCH_SIZE="${SKETCHIMAGE_TORCH_BATCH_SIZE:-128}"
export SKETCHIMAGE_TORCH_LR="${SKETCHIMAGE_TORCH_LR:-0.001}"
export SKETCHIMAGE_TORCH_WEIGHT_DECAY="${SKETCHIMAGE_TORCH_WEIGHT_DECAY:-0.0001}"
export SKETCHIMAGE_TORCH_DIFFUSION_STEPS="${SKETCHIMAGE_TORCH_DIFFUSION_STEPS:-16}"
export SKETCHIMAGE_TORCH_TRAIN_NOISE="${SKETCHIMAGE_TORCH_TRAIN_NOISE:-0.35}"
export SKETCHIMAGE_TORCH_DIRECT_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_DIRECT_LOSS_WEIGHT:-1.0}"
export SKETCHIMAGE_TORCH_DELTA_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_DELTA_LOSS_WEIGHT:-0.2}"
export SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT:-2.0}"
export SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT:-12.0}"
export SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT:-0.75}"
export SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE="${SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE:-0.04}"
export SKETCHIMAGE_TORCH_HARD_NEGATIVE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_HARD_NEGATIVE_LOSS_WEIGHT:-0.0}"
export SKETCHIMAGE_TORCH_HARD_NEGATIVE_MARGIN="${SKETCHIMAGE_TORCH_HARD_NEGATIVE_MARGIN:-0.10}"
export SKETCHIMAGE_TORCH_DEVICE="${SKETCHIMAGE_TORCH_DEVICE:-auto}"
export SKETCHIMAGE_DECODER_DEVICE="${SKETCHIMAGE_DECODER_DEVICE:-auto}"
export SKETCHIMAGE_RENDER_IMAGE_CONTEXT="${SKETCHIMAGE_RENDER_IMAGE_CONTEXT:-1}"
export SKETCHIMAGE_CALIBRATION_MODE="${SKETCHIMAGE_CALIBRATION_MODE:-residual_ridge}"
export SKETCHIMAGE_CALIBRATION_RIDGE="${SKETCHIMAGE_CALIBRATION_RIDGE:-0.01}"
export SKETCHIMAGE_CALIBRATION_BLEND="${SKETCHIMAGE_CALIBRATION_BLEND:-1.0}"
export SKETCHIMAGE_CALIBRATION_NORMALIZE="${SKETCHIMAGE_CALIBRATION_NORMALIZE:-1}"

echo "SketchImage-JEPA Phase 2C calibrated decoder"
echo "  python=$PYTHON_BIN"
echo "  run_root=$OUTPUT_DIR"
echo "  decoder_dir=$SKETCHIMAGE_DECODER_DIR"
echo "  decoder_pool_dir=$SKETCHIMAGE_DECODER_POOL_DIR"
echo "  train_csv=$SKETCHIMAGE_TRAIN_CSV"
echo "  eval_csv=$SKETCHIMAGE_EVAL_CSV"
echo "  feature_dim=$SKETCHIMAGE_FEATURE_DIM"
echo "  latent_dim=${SKETCHIMAGE_LATENT_DIM:-<decoder condition dim>}"
echo "  top_k=$SKETCHIMAGE_TOP_K"
echo "  backend=$SKETCHIMAGE_BACKEND"
echo "  torch_device=$SKETCHIMAGE_TORCH_DEVICE"
echo "  decoder_device=$SKETCHIMAGE_DECODER_DEVICE"
echo "  torch_epochs=$SKETCHIMAGE_TORCH_EPOCHS"
echo "  torch_hidden_dim=$SKETCHIMAGE_TORCH_HIDDEN_DIM"
echo "  torch_batch_size=$SKETCHIMAGE_TORCH_BATCH_SIZE"
echo "  calibration_mode=$SKETCHIMAGE_CALIBRATION_MODE"
echo "  calibration_ridge=$SKETCHIMAGE_CALIBRATION_RIDGE"
echo "  calibration_blend=$SKETCHIMAGE_CALIBRATION_BLEND"
echo "  calibration_normalize=$SKETCHIMAGE_CALIBRATION_NORMALIZE"
echo "  seed=$SKETCHIMAGE_SEED"
echo

if [[ ! -e "$SKETCHIMAGE_DECODER_DIR/model.pt" && ! -e "$SKETCHIMAGE_DECODER_DIR/model/model.pt" ]]; then
  echo "ERROR: decoder model not found under SKETCHIMAGE_DECODER_DIR=$SKETCHIMAGE_DECODER_DIR" >&2
  exit 2
fi
if [[ ! -f "$SKETCHIMAGE_TRAIN_CSV" ]]; then
  echo "ERROR: training CSV not found: $SKETCHIMAGE_TRAIN_CSV" >&2
  exit 2
fi
if [[ ! -f "$SKETCHIMAGE_EVAL_CSV" ]]; then
  echo "ERROR: eval CSV not found: $SKETCHIMAGE_EVAL_CSV" >&2
  exit 2
fi

if [[ "${SKETCHIMAGE_RUN_TESTS:-1}" == "1" ]]; then
  echo "[1/2] Running tests"
  "$PYTHON_BIN" -m unittest discover -s tests
  echo
else
  echo "[1/2] Skipping tests because SKETCHIMAGE_RUN_TESTS=$SKETCHIMAGE_RUN_TESTS"
  echo
fi

echo "[2/2] Training planner, calibrating latents, and decoding eval calibrated latents"
LATENT_ARGS=()
if [[ -n "${SKETCHIMAGE_LATENT_DIM:-}" ]]; then
  LATENT_ARGS+=(--latent-dim "$SKETCHIMAGE_LATENT_DIM")
fi
RENDER_ARGS=()
if [[ "$SKETCHIMAGE_RENDER_IMAGE_CONTEXT" == "1" ]]; then
  RENDER_ARGS+=(--render-image-context)
fi
NORMALIZE_ARGS=()
if [[ "$SKETCHIMAGE_CALIBRATION_NORMALIZE" == "1" ]]; then
  NORMALIZE_ARGS+=(--calibration-normalize)
else
  NORMALIZE_ARGS+=(--no-calibration-normalize)
fi

"$PYTHON_BIN" -m sketchimage_jepa.phase2_calibrated_decoder \
  --decoder-dir "$SKETCHIMAGE_DECODER_DIR" \
  --decoder-pool-dir "$SKETCHIMAGE_DECODER_POOL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --train-csv "$SKETCHIMAGE_TRAIN_CSV" \
  --eval-csv "$SKETCHIMAGE_EVAL_CSV" \
  --feature-dim "$SKETCHIMAGE_FEATURE_DIM" \
  "${LATENT_ARGS[@]}" \
  --top-k "$SKETCHIMAGE_TOP_K" \
  --ridge "$SKETCHIMAGE_RIDGE" \
  --backend "$SKETCHIMAGE_BACKEND" \
  --torch-hidden-dim "$SKETCHIMAGE_TORCH_HIDDEN_DIM" \
  --torch-epochs "$SKETCHIMAGE_TORCH_EPOCHS" \
  --torch-batch-size "$SKETCHIMAGE_TORCH_BATCH_SIZE" \
  --torch-lr "$SKETCHIMAGE_TORCH_LR" \
  --torch-weight-decay "$SKETCHIMAGE_TORCH_WEIGHT_DECAY" \
  --torch-diffusion-steps "$SKETCHIMAGE_TORCH_DIFFUSION_STEPS" \
  --torch-train-noise "$SKETCHIMAGE_TORCH_TRAIN_NOISE" \
  --torch-direct-loss-weight "$SKETCHIMAGE_TORCH_DIRECT_LOSS_WEIGHT" \
  --torch-delta-loss-weight "$SKETCHIMAGE_TORCH_DELTA_LOSS_WEIGHT" \
  --torch-cosine-loss-weight "$SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT" \
  --torch-positive-loss-weight "$SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT" \
  --torch-contrastive-loss-weight "$SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT" \
  --torch-contrastive-temperature "$SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE" \
  --torch-hard-negative-loss-weight "$SKETCHIMAGE_TORCH_HARD_NEGATIVE_LOSS_WEIGHT" \
  --torch-hard-negative-margin "$SKETCHIMAGE_TORCH_HARD_NEGATIVE_MARGIN" \
  --torch-device "$SKETCHIMAGE_TORCH_DEVICE" \
  --decoder-device "$SKETCHIMAGE_DECODER_DEVICE" \
  --calibration-mode "$SKETCHIMAGE_CALIBRATION_MODE" \
  --calibration-ridge "$SKETCHIMAGE_CALIBRATION_RIDGE" \
  --calibration-blend "$SKETCHIMAGE_CALIBRATION_BLEND" \
  "${NORMALIZE_ARGS[@]}" \
  --seed "$SKETCHIMAGE_SEED" \
  "${RENDER_ARGS[@]}"

echo
echo "Phase 2C calibrated decoder finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  predictions=$OUTPUT_DIR/predictions.csv"
echo "  config=$OUTPUT_DIR/run_config.json"
