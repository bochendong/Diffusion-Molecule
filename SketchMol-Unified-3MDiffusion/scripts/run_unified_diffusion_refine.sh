#!/usr/bin/env bash
# Resume Stage 3 latent diffusion, re-evaluate, and optionally materialize benchmark.
# Skips dataset export, alignment, and connector training.

set -euo pipefail

PROJECT_DIR="SketchMol-Unified-3MDiffusion"
REPO_DIR="$(pwd)"
PYTHON_BIN="${SMU3M_PYTHON_BIN:-python3}"
OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_edit_v2}"
TRAIN_JSONL="${SMU3M_TRAIN_JSONL:-$OUTPUT_DIR/dataset/unified_condition_train.jsonl}"
EVAL_JSONL="${SMU3M_EVAL_JSONL:-$OUTPUT_DIR/dataset/unified_condition_eval.jsonl}"
CONNECTOR="${SMU3M_CONDITION_CONNECTOR:-$OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt}"
DIFFUSION_DIR="${SMU3M_DIFFUSION_DIR:-$OUTPUT_DIR/latent_diffusion}"
EVAL_DIR="${SMU3M_EVAL_LATENT_DIR:-$OUTPUT_DIR/eval_latent}"

TRAIN_LIMIT="${SMU3M_TRAIN_LIMIT:-50000}"
BATCH_SIZE="${SMU3M_BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${SMU3M_EVAL_BATCH_SIZE:-512}"
NUM_WORKERS="${SMU3M_NUM_WORKERS:-0}"
PIN_MEMORY="${SMU3M_PIN_MEMORY:-0}"
DIFFUSION_EPOCHS="${SMU3M_DIFFUSION_EPOCHS:-150}"
DIFFUSION_LR="${SMU3M_DIFFUSION_LR:-3e-4}"
EVAL_LIMIT="${SMU3M_EVAL_LIMIT:-1000}"
EVAL_SAMPLE_STEPS="${SMU3M_EVAL_SAMPLE_STEPS:-20}"
EVAL_SAMPLE_ETA="${SMU3M_EVAL_SAMPLE_ETA:-0.0}"
DIFFUSION_TIMESTEPS="${SMU3M_DIFFUSION_TIMESTEPS:-100}"
DIFFUSION_OBJECTIVE="${SMU3M_DIFFUSION_OBJECTIVE:-pred_x0}"
DIFFUSION_TARGET="${SMU3M_DIFFUSION_TARGET:-residual}"
PRIOR_LOSS_WEIGHT="${SMU3M_PRIOR_LOSS_WEIGHT:-0.0}"
DIFFUSION_HIDDEN_DIM="${SMU3M_DIFFUSION_HIDDEN_DIM:-512}"
DIFFUSION_DEPTH="${SMU3M_DIFFUSION_DEPTH:-4}"
CHECKPOINT_EVERY="${SMU3M_CHECKPOINT_EVERY:-1}"
RESUME="${SMU3M_RESUME:-1}"
REQUIRE_CUDA="${SMU3M_REQUIRE_CUDA:-1}"
DEVICE="${SMU3M_DEVICE:-auto}"
TRAIN_DIFFUSION_CONNECTOR="${SMU3M_TRAIN_DIFFUSION_CONNECTOR:-0}"
RUN_MATERIALIZED_BENCHMARK="${SMU3M_RUN_MATERIALIZED_BENCHMARK:-1}"

echo "Running Unified 3M Stage 3 diffusion refine"
echo "  python=$PYTHON_BIN"
echo "  output_dir=$OUTPUT_DIR"
echo "  diffusion_dir=$DIFFUSION_DIR"
echo "  diffusion_epochs=$DIFFUSION_EPOCHS"
echo "  diffusion_lr=$DIFFUSION_LR"
echo "  train_diffusion_connector=$TRAIN_DIFFUSION_CONNECTOR"
echo "  eval_limit=$EVAL_LIMIT"
echo "  run_materialized_benchmark=$RUN_MATERIALIZED_BENCHMARK"

for required in "$TRAIN_JSONL" "$EVAL_JSONL" "$CONNECTOR"; do
  if [ ! -f "$required" ]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done

if [ "$REQUIRE_CUDA" = "1" ]; then
  "$PYTHON_BIN" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA is required for diffusion refine.", file=sys.stderr)
    sys.exit(2)
PY
fi

DIFFUSION_ARGS=(
  --train-jsonl "$TRAIN_JSONL"
  --condition-connector "$CONNECTOR"
  --output-dir "$DIFFUSION_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --epochs "$DIFFUSION_EPOCHS"
  --limit "$TRAIN_LIMIT"
  --timesteps "$DIFFUSION_TIMESTEPS"
  --diffusion-objective "$DIFFUSION_OBJECTIVE"
  --diffusion-target "$DIFFUSION_TARGET"
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT"
  --lr "$DIFFUSION_LR"
  --hidden-dim "$DIFFUSION_HIDDEN_DIM"
  --depth "$DIFFUSION_DEPTH"
  --device "$DEVICE"
  --checkpoint-every "$CHECKPOINT_EVERY"
)
if [ "$TRAIN_DIFFUSION_CONNECTOR" = "1" ]; then
  DIFFUSION_ARGS+=(--train-connector)
fi
if [ "$PIN_MEMORY" = "1" ]; then
  DIFFUSION_ARGS+=(--pin-memory)
fi
if [ "$RESUME" = "1" ] && [ -f "$DIFFUSION_DIR/checkpoints/latest.pt" ]; then
  if [ -f "$DIFFUSION_DIR/metrics.json" ] \
    && grep -q "\"diffusion_objective\": \"$DIFFUSION_OBJECTIVE\"" "$DIFFUSION_DIR/metrics.json" \
    && grep -q "\"diffusion_target\": \"$DIFFUSION_TARGET\"" "$DIFFUSION_DIR/metrics.json"; then
    DIFFUSION_ARGS+=(--resume-checkpoint "$DIFFUSION_DIR/checkpoints/latest.pt")
  else
    echo "Existing latent diffusion checkpoint predates $DIFFUSION_OBJECTIVE/$DIFFUSION_TARGET settings; retraining latent diffusion."
  fi
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_latent_diffusion_generation.py" "${DIFFUSION_ARGS[@]}"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_latent_diffusion_generation.py" \
  --eval-jsonl "$EVAL_JSONL" \
  --condition-connector "$CONNECTOR" \
  --diffusion-checkpoint "$DIFFUSION_DIR/latent_diffusion_generation.pt" \
  --output-dir "$EVAL_DIR" \
  --limit "$EVAL_LIMIT" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --sample-steps "$EVAL_SAMPLE_STEPS" \
  --sample-eta "$EVAL_SAMPLE_ETA" \
  --device "$DEVICE"

if [ "$RUN_MATERIALIZED_BENCHMARK" = "1" ]; then
  rm -f \
    "$EVAL_DIR/edit_latent_predictions.npy" \
    "$EVAL_DIR/edit_latent_fingerprints.npy" \
    "$EVAL_DIR/index.csv" \
    "$EVAL_DIR/benchmark_export_metrics.json"
  SMU3M_OUTPUT_DIR="$OUTPUT_DIR" \
  SMU3M_EVAL_LATENT_DIR="$EVAL_DIR" \
  SMU3M_PYTHON_BIN="$PYTHON_BIN" \
    bash "$PROJECT_DIR/scripts/run_unified_materialized_benchmark.sh"
fi

echo "Unified diffusion refine finished:"
echo "  diffusion=$DIFFUSION_DIR/latent_diffusion_generation.pt"
echo "  eval=$EVAL_DIR/metrics.json"
if [ "$RUN_MATERIALIZED_BENCHMARK" = "1" ]; then
  echo "  benchmark=$OUTPUT_DIR/benchmark_materialized/benchmark_report.md"
fi
