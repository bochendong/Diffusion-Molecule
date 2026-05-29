#!/usr/bin/env bash
# Run Phase 2D: oracle-anchored robust decoder fine-tuning.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${SKETCHIMAGE_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
RUN_NAME="${SKETCHIMAGE_RUN_NAME:-phase2_oracle_anchored_decoder_2048_seed${SKETCHIMAGE_SEED:-7}}"
OUTPUT_DIR="${SKETCHIMAGE_OUTPUT_DIR:-outputs/runs/$RUN_NAME}"

export SKETCHIMAGE_SEED="${SKETCHIMAGE_SEED:-7}"
export SKETCHIMAGE_ORACLE_DECODER_DIR="${SKETCHIMAGE_ORACLE_DECODER_DIR:-outputs/runs/phase1_oracle_latent_ar_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_DECODER_POOL_DIR="${SKETCHIMAGE_DECODER_POOL_DIR:-outputs/runs/phase1_oracle_latent_ar_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_PLANNER_RUN_DIR="${SKETCHIMAGE_PLANNER_RUN_DIR:-outputs/runs/phase2_robust_decoder_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_CALIBRATED_RUN_DIR="${SKETCHIMAGE_CALIBRATED_RUN_DIR:-outputs/runs/phase2_calibrated_decoder_2048_seed${SKETCHIMAGE_SEED}}"
export SKETCHIMAGE_TRAIN_CSV="${SKETCHIMAGE_TRAIN_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED}_train.csv}"
export SKETCHIMAGE_EVAL_CSV="${SKETCHIMAGE_EVAL_CSV:-outputs/tasks/sketchmol_hard_seed${SKETCHIMAGE_SEED}_eval.csv}"
export SKETCHIMAGE_FEATURE_DIM="${SKETCHIMAGE_FEATURE_DIM:-256}"
export SKETCHIMAGE_TOP_K="${SKETCHIMAGE_TOP_K:-8}"
export SKETCHIMAGE_DECODER_DEVICE="${SKETCHIMAGE_DECODER_DEVICE:-auto}"
export SKETCHIMAGE_DECODER_FINETUNE_EPOCHS="${SKETCHIMAGE_DECODER_FINETUNE_EPOCHS:-4}"
export SKETCHIMAGE_DECODER_FINETUNE_BATCH_SIZE="${SKETCHIMAGE_DECODER_FINETUNE_BATCH_SIZE:-128}"
export SKETCHIMAGE_DECODER_FINETUNE_LR="${SKETCHIMAGE_DECODER_FINETUNE_LR:-0.00002}"
export SKETCHIMAGE_DECODER_FINETUNE_WEIGHT_DECAY="${SKETCHIMAGE_DECODER_FINETUNE_WEIGHT_DECAY:-0.0001}"
export SKETCHIMAGE_DECODER_ORACLE_REPEATS="${SKETCHIMAGE_DECODER_ORACLE_REPEATS:-8}"
export SKETCHIMAGE_DECODER_NOISY_REPEATS="${SKETCHIMAGE_DECODER_NOISY_REPEATS:-1}"
export SKETCHIMAGE_DECODER_NOISY_COSINES="${SKETCHIMAGE_DECODER_NOISY_COSINES:-0.78,0.90}"
export SKETCHIMAGE_DECODER_PLANNER_REPEATS="${SKETCHIMAGE_DECODER_PLANNER_REPEATS:-1}"
export SKETCHIMAGE_DECODER_CALIBRATED_REPEATS="${SKETCHIMAGE_DECODER_CALIBRATED_REPEATS:-1}"
export SKETCHIMAGE_DECODER_INTERPOLATION_REPEATS="${SKETCHIMAGE_DECODER_INTERPOLATION_REPEATS:-1}"
export SKETCHIMAGE_DECODER_INTERPOLATION_ALPHAS="${SKETCHIMAGE_DECODER_INTERPOLATION_ALPHAS:-0.10,0.25}"

echo "SketchImage-JEPA Phase 2D oracle-anchored decoder"
echo "  python=$PYTHON_BIN"
echo "  run_root=$OUTPUT_DIR"
echo "  oracle_decoder_dir=$SKETCHIMAGE_ORACLE_DECODER_DIR"
echo "  decoder_pool_dir=$SKETCHIMAGE_DECODER_POOL_DIR"
echo "  planner_run_dir=$SKETCHIMAGE_PLANNER_RUN_DIR"
echo "  calibrated_run_dir=$SKETCHIMAGE_CALIBRATED_RUN_DIR"
echo "  train_csv=$SKETCHIMAGE_TRAIN_CSV"
echo "  eval_csv=$SKETCHIMAGE_EVAL_CSV"
echo "  feature_dim=$SKETCHIMAGE_FEATURE_DIM"
echo "  latent_dim=${SKETCHIMAGE_LATENT_DIM:-<decoder condition dim>}"
echo "  top_k=$SKETCHIMAGE_TOP_K"
echo "  decoder_device=$SKETCHIMAGE_DECODER_DEVICE"
echo "  decoder_finetune_epochs=$SKETCHIMAGE_DECODER_FINETUNE_EPOCHS"
echo "  decoder_finetune_lr=$SKETCHIMAGE_DECODER_FINETUNE_LR"
echo "  decoder_oracle_repeats=$SKETCHIMAGE_DECODER_ORACLE_REPEATS"
echo "  decoder_noisy_repeats=$SKETCHIMAGE_DECODER_NOISY_REPEATS"
echo "  decoder_noisy_cosines=$SKETCHIMAGE_DECODER_NOISY_COSINES"
echo "  decoder_planner_repeats=$SKETCHIMAGE_DECODER_PLANNER_REPEATS"
echo "  decoder_calibrated_repeats=$SKETCHIMAGE_DECODER_CALIBRATED_REPEATS"
echo "  decoder_interpolation_repeats=$SKETCHIMAGE_DECODER_INTERPOLATION_REPEATS"
echo "  decoder_interpolation_alphas=$SKETCHIMAGE_DECODER_INTERPOLATION_ALPHAS"
echo "  seed=$SKETCHIMAGE_SEED"
echo

if [[ ! -e "$SKETCHIMAGE_ORACLE_DECODER_DIR/model.pt" && ! -e "$SKETCHIMAGE_ORACLE_DECODER_DIR/model/model.pt" ]]; then
  echo "ERROR: oracle decoder model not found under SKETCHIMAGE_ORACLE_DECODER_DIR=$SKETCHIMAGE_ORACLE_DECODER_DIR" >&2
  exit 2
fi
if [[ ! -f "$SKETCHIMAGE_PLANNER_RUN_DIR/planner_train_latents.npy" || ! -f "$SKETCHIMAGE_PLANNER_RUN_DIR/planner_eval_latents.npy" ]]; then
  echo "ERROR: planner latent artifacts not found under $SKETCHIMAGE_PLANNER_RUN_DIR" >&2
  exit 2
fi
if [[ ! -f "$SKETCHIMAGE_CALIBRATED_RUN_DIR/calibrated_train_latents.npy" || ! -f "$SKETCHIMAGE_CALIBRATED_RUN_DIR/calibrated_eval_latents.npy" ]]; then
  echo "ERROR: calibrated latent artifacts not found under $SKETCHIMAGE_CALIBRATED_RUN_DIR" >&2
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

echo "[2/2] Fine-tuning oracle-anchored decoder and evaluating sources"
LATENT_ARGS=()
if [[ -n "${SKETCHIMAGE_LATENT_DIM:-}" ]]; then
  LATENT_ARGS+=(--latent-dim "$SKETCHIMAGE_LATENT_DIM")
fi

"$PYTHON_BIN" -m sketchimage_jepa.phase2_oracle_anchored_decoder \
  --oracle-decoder-dir "$SKETCHIMAGE_ORACLE_DECODER_DIR" \
  --decoder-pool-dir "$SKETCHIMAGE_DECODER_POOL_DIR" \
  --planner-run-dir "$SKETCHIMAGE_PLANNER_RUN_DIR" \
  --calibrated-run-dir "$SKETCHIMAGE_CALIBRATED_RUN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --train-csv "$SKETCHIMAGE_TRAIN_CSV" \
  --eval-csv "$SKETCHIMAGE_EVAL_CSV" \
  --feature-dim "$SKETCHIMAGE_FEATURE_DIM" \
  "${LATENT_ARGS[@]}" \
  --top-k "$SKETCHIMAGE_TOP_K" \
  --decoder-device "$SKETCHIMAGE_DECODER_DEVICE" \
  --decoder-finetune-epochs "$SKETCHIMAGE_DECODER_FINETUNE_EPOCHS" \
  --decoder-finetune-batch-size "$SKETCHIMAGE_DECODER_FINETUNE_BATCH_SIZE" \
  --decoder-finetune-lr "$SKETCHIMAGE_DECODER_FINETUNE_LR" \
  --decoder-finetune-weight-decay "$SKETCHIMAGE_DECODER_FINETUNE_WEIGHT_DECAY" \
  --decoder-oracle-repeats "$SKETCHIMAGE_DECODER_ORACLE_REPEATS" \
  --decoder-noisy-repeats "$SKETCHIMAGE_DECODER_NOISY_REPEATS" \
  --decoder-noisy-cosines "$SKETCHIMAGE_DECODER_NOISY_COSINES" \
  --decoder-planner-repeats "$SKETCHIMAGE_DECODER_PLANNER_REPEATS" \
  --decoder-calibrated-repeats "$SKETCHIMAGE_DECODER_CALIBRATED_REPEATS" \
  --decoder-interpolation-repeats "$SKETCHIMAGE_DECODER_INTERPOLATION_REPEATS" \
  --decoder-interpolation-alphas "$SKETCHIMAGE_DECODER_INTERPOLATION_ALPHAS" \
  --seed "$SKETCHIMAGE_SEED"

echo
echo "Phase 2D oracle-anchored decoder finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  predictions=$OUTPUT_DIR/predictions.csv"
echo "  source_summary=$OUTPUT_DIR/source_summary.csv"
echo "  config=$OUTPUT_DIR/run_config.json"
