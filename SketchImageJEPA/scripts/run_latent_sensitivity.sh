#!/usr/bin/env bash
# Diagnose decoder sensitivity to oracle, noisy, planner, and calibrated latents.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
RUN_NAME="${SKETCHIMAGE_RUN_NAME:-phase2_latent_sensitivity_2048_seed${SKETCHIMAGE_SEED:-7}}"
OUTPUT_DIR="${SKETCHIMAGE_OUTPUT_DIR:-outputs/runs/$RUN_NAME}"

export SKETCHIMAGE_SEED="${SKETCHIMAGE_SEED:-7}"
export SKETCHIMAGE_DECODER_DIR="${SKETCHIMAGE_DECODER_DIR:-outputs/runs/phase2_robust_decoder_2048_seed${SKETCHIMAGE_SEED}/decoder}"
export SKETCHIMAGE_DECODER_POOL_DIR="${SKETCHIMAGE_DECODER_POOL_DIR:-outputs/runs/phase1_oracle_latent_ar_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_PLANNER_RUN_DIR="${SKETCHIMAGE_PLANNER_RUN_DIR:-outputs/runs/phase2_robust_decoder_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_CALIBRATED_RUN_DIR="${SKETCHIMAGE_CALIBRATED_RUN_DIR:-outputs/runs/phase2_calibrated_decoder_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_TRAIN_CSV="${SKETCHIMAGE_TRAIN_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED}_train.csv}"
export SKETCHIMAGE_EVAL_CSV="${SKETCHIMAGE_EVAL_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED}_eval.csv}"
export SKETCHIMAGE_FEATURE_DIM="${SKETCHIMAGE_FEATURE_DIM:-256}"
export SKETCHIMAGE_TOP_K="${SKETCHIMAGE_TOP_K:-8}"
export SKETCHIMAGE_NOISY_COSINES="${SKETCHIMAGE_NOISY_COSINES:-0.32,0.38,0.63,0.78}"
export SKETCHIMAGE_INTERPOLATION_ALPHAS="${SKETCHIMAGE_INTERPOLATION_ALPHAS:-0.25,0.50,0.75}"
export SKETCHIMAGE_DECODER_DEVICE="${SKETCHIMAGE_DECODER_DEVICE:-auto}"

echo "SketchImage-JEPA latent sensitivity diagnostic"
echo "  python=$PYTHON_BIN"
echo "  run_root=$OUTPUT_DIR"
echo "  decoder_dir=$SKETCHIMAGE_DECODER_DIR"
echo "  decoder_pool_dir=$SKETCHIMAGE_DECODER_POOL_DIR"
echo "  planner_run_dir=$SKETCHIMAGE_PLANNER_RUN_DIR"
echo "  calibrated_run_dir=$SKETCHIMAGE_CALIBRATED_RUN_DIR"
echo "  train_csv=$SKETCHIMAGE_TRAIN_CSV"
echo "  eval_csv=$SKETCHIMAGE_EVAL_CSV"
echo "  feature_dim=$SKETCHIMAGE_FEATURE_DIM"
echo "  latent_dim=${SKETCHIMAGE_LATENT_DIM:-<decoder condition dim>}"
echo "  top_k=$SKETCHIMAGE_TOP_K"
echo "  noisy_cosines=$SKETCHIMAGE_NOISY_COSINES"
echo "  interpolation_alphas=$SKETCHIMAGE_INTERPOLATION_ALPHAS"
echo "  max_eval_tasks=${SKETCHIMAGE_MAX_EVAL_TASKS:-<all>}"
echo "  decoder_device=$SKETCHIMAGE_DECODER_DEVICE"
echo "  seed=$SKETCHIMAGE_SEED"
echo

if [[ ! -e "$SKETCHIMAGE_DECODER_DIR/model.pt" && ! -e "$SKETCHIMAGE_DECODER_DIR/model/model.pt" ]]; then
  echo "ERROR: decoder model not found under SKETCHIMAGE_DECODER_DIR=$SKETCHIMAGE_DECODER_DIR" >&2
  exit 2
fi
if [[ ! -f "$SKETCHIMAGE_PLANNER_RUN_DIR/planner_eval_latents.npy" ]]; then
  echo "ERROR: planner eval latents not found: $SKETCHIMAGE_PLANNER_RUN_DIR/planner_eval_latents.npy" >&2
  exit 2
fi
if [[ ! -f "$SKETCHIMAGE_CALIBRATED_RUN_DIR/calibrated_eval_latents.npy" ]]; then
  echo "ERROR: calibrated eval latents not found: $SKETCHIMAGE_CALIBRATED_RUN_DIR/calibrated_eval_latents.npy" >&2
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

echo "[2/2] Decoding controlled latent sources"
LATENT_ARGS=()
if [[ -n "${SKETCHIMAGE_LATENT_DIM:-}" ]]; then
  LATENT_ARGS+=(--latent-dim "$SKETCHIMAGE_LATENT_DIM")
fi
MAX_EVAL_ARGS=()
if [[ -n "${SKETCHIMAGE_MAX_EVAL_TASKS:-}" ]]; then
  MAX_EVAL_ARGS+=(--max-eval-tasks "$SKETCHIMAGE_MAX_EVAL_TASKS")
fi

"$PYTHON_BIN" -m sketchimage_jepa.latent_sensitivity \
  --decoder-dir "$SKETCHIMAGE_DECODER_DIR" \
  --decoder-pool-dir "$SKETCHIMAGE_DECODER_POOL_DIR" \
  --planner-run-dir "$SKETCHIMAGE_PLANNER_RUN_DIR" \
  --calibrated-run-dir "$SKETCHIMAGE_CALIBRATED_RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --train-csv "$SKETCHIMAGE_TRAIN_CSV" \
  --eval-csv "$SKETCHIMAGE_EVAL_CSV" \
  --feature-dim "$SKETCHIMAGE_FEATURE_DIM" \
  "${LATENT_ARGS[@]}" \
  --top-k "$SKETCHIMAGE_TOP_K" \
  --noisy-cosines "$SKETCHIMAGE_NOISY_COSINES" \
  --interpolation-alphas "$SKETCHIMAGE_INTERPOLATION_ALPHAS" \
  --decoder-device "$SKETCHIMAGE_DECODER_DEVICE" \
  --seed "$SKETCHIMAGE_SEED" \
  "${MAX_EVAL_ARGS[@]}"

echo
echo "Latent sensitivity diagnostic finished: $OUTPUT_DIR"
echo "  summary=$OUTPUT_DIR/source_summary.csv"
echo "  summary_json=$OUTPUT_DIR/source_summary.json"
echo "  config=$OUTPUT_DIR/run_config.json"
