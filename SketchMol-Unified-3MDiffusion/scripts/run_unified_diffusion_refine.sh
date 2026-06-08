#!/usr/bin/env bash
# Resume Stage 3 latent diffusion and re-evaluate latent metrics.
# Skips dataset export, alignment, and connector training.

set -euo pipefail

PROJECT_DIR="SketchMol-Unified-3MDiffusion"
REPO_DIR="$(pwd)"
PYTHON_BIN="${SMU3M_PYTHON_BIN:-python3}"
OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_3m_source_neighbor_sourceguard_v1}"
TRAIN_JSONL="${SMU3M_TRAIN_JSONL:-$OUTPUT_DIR/dataset/unified_condition_train.jsonl}"
EVAL_JSONL="${SMU3M_EVAL_JSONL:-$OUTPUT_DIR/dataset/unified_condition_eval.jsonl}"
CONNECTOR="${SMU3M_CONDITION_CONNECTOR:-$OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt}"
DIFFUSION_DIR="${SMU3M_DIFFUSION_DIR:-$OUTPUT_DIR/latent_diffusion}"
BASE_DIFFUSION_DIR="${SMU3M_BASE_DIFFUSION_DIR:-$OUTPUT_DIR/latent_diffusion}"
RESUME_DIFFUSION_CHECKPOINT="${SMU3M_RESUME_DIFFUSION_CHECKPOINT:-}"
EVAL_DIR="${SMU3M_EVAL_LATENT_DIR:-$OUTPUT_DIR/eval_latent}"

TRAIN_LIMIT="${SMU3M_TRAIN_LIMIT:-50000}"
BATCH_SIZE="${SMU3M_BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${SMU3M_EVAL_BATCH_SIZE:-512}"
NUM_WORKERS="${SMU3M_NUM_WORKERS:-0}"
PIN_MEMORY="${SMU3M_PIN_MEMORY:-0}"
DIFFUSION_EPOCHS="${SMU3M_DIFFUSION_EPOCHS:-}"
DIFFUSION_EXTRA_EPOCHS="${SMU3M_DIFFUSION_EXTRA_EPOCHS:-100}"
DIFFUSION_LR="${SMU3M_DIFFUSION_LR:-3e-4}"
DIFFUSION_SEED="${SMU3M_DIFFUSION_SEED:-13}"
EVAL_SEED="${SMU3M_EVAL_SEED:-17}"
EVAL_LIMIT="${SMU3M_EVAL_LIMIT:-0}"
MAX_EVAL_PER_PROPERTY_COUNT="${SMU3M_MAX_EVAL_PER_PROPERTY_COUNT:-250}"
EVAL_SAMPLE_STEPS="${SMU3M_EVAL_SAMPLE_STEPS:-20}"
EVAL_SAMPLE_ETA="${SMU3M_EVAL_SAMPLE_ETA:-0.0}"
DIFFUSION_TIMESTEPS="${SMU3M_DIFFUSION_TIMESTEPS:-100}"
DIFFUSION_OBJECTIVE="${SMU3M_DIFFUSION_OBJECTIVE:-pred_x0}"
DIFFUSION_TARGET="${SMU3M_DIFFUSION_TARGET:-residual}"
PRIOR_LOSS_WEIGHT="${SMU3M_PRIOR_LOSS_WEIGHT:-0.25}"
SOURCE_REGRET_LOSS_WEIGHT="${SMU3M_SOURCE_REGRET_LOSS_WEIGHT:-0.35}"
SOURCE_REGRET_MARGIN="${SMU3M_SOURCE_REGRET_MARGIN:-0.0}"
SOURCE_RADIUS_LOSS_WEIGHT="${SMU3M_SOURCE_RADIUS_LOSS_WEIGHT:-0.10}"
SOURCE_RADIUS_MARGIN="${SMU3M_SOURCE_RADIUS_MARGIN:-0.05}"
SOURCE_SIMILARITY_WEIGHT_FLOOR="${SMU3M_SOURCE_SIMILARITY_WEIGHT_FLOOR:-0.25}"
SOURCE_FINGERPRINT_PRIOR_BLEND="${SMU3M_SOURCE_FINGERPRINT_PRIOR_BLEND:-0.85}"
FINGERPRINT_GUARD_LOSS_WEIGHT="${SMU3M_FINGERPRINT_GUARD_LOSS_WEIGHT:-0.50}"
FINGERPRINT_GUARD_MARGIN="${SMU3M_FINGERPRINT_GUARD_MARGIN:-0.02}"
DIFFUSION_HIDDEN_DIM="${SMU3M_DIFFUSION_HIDDEN_DIM:-512}"
DIFFUSION_DEPTH="${SMU3M_DIFFUSION_DEPTH:-4}"
CHECKPOINT_EVERY="${SMU3M_CHECKPOINT_EVERY:-1}"
RESUME="${SMU3M_RESUME:-1}"
ALLOW_INCOMPATIBLE_RESUME_WEIGHTS="${SMU3M_ALLOW_INCOMPATIBLE_RESUME_WEIGHTS:-0}"
REQUIRE_CUDA="${SMU3M_REQUIRE_CUDA:-1}"
DEVICE="${SMU3M_DEVICE:-auto}"
TRAIN_DIFFUSION_CONNECTOR="${SMU3M_TRAIN_DIFFUSION_CONNECTOR:-1}"
RUN_MATERIALIZED_BENCHMARK="${SMU3M_RUN_MATERIALIZED_BENCHMARK:-0}"

if [ "$RESUME" = "1" ] && [ -z "$RESUME_DIFFUSION_CHECKPOINT" ]; then
  if [ -f "$DIFFUSION_DIR/checkpoints/latest.pt" ]; then
    RESUME_DIFFUSION_CHECKPOINT="$DIFFUSION_DIR/checkpoints/latest.pt"
  elif [ -f "$BASE_DIFFUSION_DIR/checkpoints/latest.pt" ]; then
    RESUME_DIFFUSION_CHECKPOINT="$BASE_DIFFUSION_DIR/checkpoints/latest.pt"
  fi
fi

if [ -z "$DIFFUSION_EPOCHS" ]; then
  RESUME_EPOCH_CHECKPOINT="${RESUME_DIFFUSION_CHECKPOINT:-__none__}"
  DIFFUSION_EPOCHS="$("$PYTHON_BIN" - "$RESUME_EPOCH_CHECKPOINT" "$DIFFUSION_EXTRA_EPOCHS" "$DIFFUSION_OBJECTIVE" "$DIFFUSION_TARGET" "$PRIOR_LOSS_WEIGHT" "$SOURCE_REGRET_LOSS_WEIGHT" "$SOURCE_REGRET_MARGIN" "$SOURCE_RADIUS_LOSS_WEIGHT" "$SOURCE_RADIUS_MARGIN" "$SOURCE_SIMILARITY_WEIGHT_FLOOR" "$SOURCE_FINGERPRINT_PRIOR_BLEND" "$FINGERPRINT_GUARD_LOSS_WEIGHT" "$FINGERPRINT_GUARD_MARGIN" "$TRAIN_DIFFUSION_CONNECTOR" <<'PY'
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
checkpoint_arg = sys.argv[1]
extra_epochs = int(float(sys.argv[2]))
objective = sys.argv[3]
target = sys.argv[4]
prior_weight = float(sys.argv[5])
source_regret_weight = float(sys.argv[6])
source_regret_margin = float(sys.argv[7])
source_radius_weight = float(sys.argv[8])
source_radius_margin = float(sys.argv[9])
source_similarity_weight_floor = float(sys.argv[10])
source_fingerprint_prior_blend = float(sys.argv[11])
fingerprint_guard_weight = float(sys.argv[12])
fingerprint_guard_margin = float(sys.argv[13])
train_connector = sys.argv[14] == "1"
checkpoint = Path(checkpoint_arg)
if checkpoint_arg != "__none__" and checkpoint.exists():
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload.get("config", {})
    compatible = (
        isinstance(config, dict)
        and str(config.get("diffusion_objective", "pred_noise")) == objective
        and str(config.get("diffusion_target", "target")) == target
        and abs(float(config.get("prior_loss_weight", 0.0)) - prior_weight) <= 1e-12
        and abs(float(config.get("source_regret_loss_weight", 0.0)) - source_regret_weight) <= 1e-12
        and abs(float(config.get("source_regret_margin", 0.0)) - source_regret_margin) <= 1e-12
        and abs(float(config.get("source_radius_loss_weight", 0.0)) - source_radius_weight) <= 1e-12
        and abs(float(config.get("source_radius_margin", 0.05)) - source_radius_margin) <= 1e-12
        and abs(float(config.get("source_similarity_weight_floor", 0.25)) - source_similarity_weight_floor) <= 1e-12
        and abs(float(config.get("source_fingerprint_prior_blend", 0.0)) - source_fingerprint_prior_blend) <= 1e-12
        and abs(float(config.get("fingerprint_guard_loss_weight", 0.0)) - fingerprint_guard_weight) <= 1e-12
        and abs(float(config.get("fingerprint_guard_margin", 0.02)) - fingerprint_guard_margin) <= 1e-12
        and bool(config.get("train_connector", False)) == train_connector
    )
    start_epoch = int(payload.get("epoch", 0)) if compatible else 0
else:
    start_epoch = 0
print(start_epoch + max(1, extra_epochs))
PY
)"
fi

echo "Running Unified 3M Stage 3 diffusion refine"
echo "  python=$PYTHON_BIN"
echo "  output_dir=$OUTPUT_DIR"
echo "  diffusion_dir=$DIFFUSION_DIR"
echo "  base_diffusion_dir=$BASE_DIFFUSION_DIR"
echo "  resume_diffusion_checkpoint=$RESUME_DIFFUSION_CHECKPOINT"
echo "  diffusion_epochs=$DIFFUSION_EPOCHS"
echo "  diffusion_extra_epochs=$DIFFUSION_EXTRA_EPOCHS"
echo "  diffusion_lr=$DIFFUSION_LR"
echo "  diffusion_seed=$DIFFUSION_SEED"
echo "  eval_seed=$EVAL_SEED"
echo "  prior_loss_weight=$PRIOR_LOSS_WEIGHT"
echo "  source_regret_loss_weight=$SOURCE_REGRET_LOSS_WEIGHT"
echo "  source_regret_margin=$SOURCE_REGRET_MARGIN"
echo "  source_radius_loss_weight=$SOURCE_RADIUS_LOSS_WEIGHT"
echo "  source_radius_margin=$SOURCE_RADIUS_MARGIN"
echo "  source_similarity_weight_floor=$SOURCE_SIMILARITY_WEIGHT_FLOOR"
echo "  source_fingerprint_prior_blend=$SOURCE_FINGERPRINT_PRIOR_BLEND"
echo "  fingerprint_guard_loss_weight=$FINGERPRINT_GUARD_LOSS_WEIGHT"
echo "  fingerprint_guard_margin=$FINGERPRINT_GUARD_MARGIN"
echo "  train_diffusion_connector=$TRAIN_DIFFUSION_CONNECTOR"
echo "  allow_incompatible_resume_weights=$ALLOW_INCOMPATIBLE_RESUME_WEIGHTS"
echo "  eval_limit=$EVAL_LIMIT"
echo "  max_eval_per_property_count=$MAX_EVAL_PER_PROPERTY_COUNT"
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
  --source-regret-loss-weight "$SOURCE_REGRET_LOSS_WEIGHT"
  --source-regret-margin "$SOURCE_REGRET_MARGIN"
  --source-radius-loss-weight "$SOURCE_RADIUS_LOSS_WEIGHT"
  --source-radius-margin "$SOURCE_RADIUS_MARGIN"
  --source-similarity-weight-floor "$SOURCE_SIMILARITY_WEIGHT_FLOOR"
  --source-fingerprint-prior-blend "$SOURCE_FINGERPRINT_PRIOR_BLEND"
  --fingerprint-guard-loss-weight "$FINGERPRINT_GUARD_LOSS_WEIGHT"
  --fingerprint-guard-margin "$FINGERPRINT_GUARD_MARGIN"
  --lr "$DIFFUSION_LR"
  --hidden-dim "$DIFFUSION_HIDDEN_DIM"
  --depth "$DIFFUSION_DEPTH"
  --device "$DEVICE"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --seed "$DIFFUSION_SEED"
)
if [ "$TRAIN_DIFFUSION_CONNECTOR" = "1" ]; then
  DIFFUSION_ARGS+=(--train-connector)
fi
if [ "$ALLOW_INCOMPATIBLE_RESUME_WEIGHTS" = "1" ]; then
  DIFFUSION_ARGS+=(--allow-incompatible-resume-weights)
fi
if [ "$PIN_MEMORY" = "1" ]; then
  DIFFUSION_ARGS+=(--pin-memory)
fi
if [ "$RESUME" = "1" ] && [ -n "$RESUME_DIFFUSION_CHECKPOINT" ] && [ -f "$RESUME_DIFFUSION_CHECKPOINT" ]; then
  if "$PYTHON_BIN" - "$RESUME_DIFFUSION_CHECKPOINT" "$DIFFUSION_OBJECTIVE" "$DIFFUSION_TARGET" "$PRIOR_LOSS_WEIGHT" "$SOURCE_REGRET_LOSS_WEIGHT" "$SOURCE_REGRET_MARGIN" "$SOURCE_RADIUS_LOSS_WEIGHT" "$SOURCE_RADIUS_MARGIN" "$SOURCE_SIMILARITY_WEIGHT_FLOOR" "$SOURCE_FINGERPRINT_PRIOR_BLEND" "$FINGERPRINT_GUARD_LOSS_WEIGHT" "$FINGERPRINT_GUARD_MARGIN" "$TRAIN_DIFFUSION_CONNECTOR" <<'PY'
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
checkpoint = Path(sys.argv[1])
objective = sys.argv[2]
target = sys.argv[3]
prior_weight = float(sys.argv[4])
source_regret_weight = float(sys.argv[5])
source_regret_margin = float(sys.argv[6])
source_radius_weight = float(sys.argv[7])
source_radius_margin = float(sys.argv[8])
source_similarity_weight_floor = float(sys.argv[9])
source_fingerprint_prior_blend = float(sys.argv[10])
fingerprint_guard_weight = float(sys.argv[11])
fingerprint_guard_margin = float(sys.argv[12])
train_connector = sys.argv[13] == "1"
payload = torch.load(checkpoint, map_location="cpu")
config = payload.get("config", {})
ok = (
    isinstance(config, dict)
    and str(config.get("diffusion_objective", "pred_noise")) == objective
    and str(config.get("diffusion_target", "target")) == target
    and abs(float(config.get("prior_loss_weight", 0.0)) - prior_weight) <= 1e-12
    and abs(float(config.get("source_regret_loss_weight", 0.0)) - source_regret_weight) <= 1e-12
    and abs(float(config.get("source_regret_margin", 0.0)) - source_regret_margin) <= 1e-12
    and abs(float(config.get("source_radius_loss_weight", 0.0)) - source_radius_weight) <= 1e-12
    and abs(float(config.get("source_radius_margin", 0.05)) - source_radius_margin) <= 1e-12
    and abs(float(config.get("source_similarity_weight_floor", 0.25)) - source_similarity_weight_floor) <= 1e-12
    and abs(float(config.get("source_fingerprint_prior_blend", 0.0)) - source_fingerprint_prior_blend) <= 1e-12
    and abs(float(config.get("fingerprint_guard_loss_weight", 0.0)) - fingerprint_guard_weight) <= 1e-12
    and abs(float(config.get("fingerprint_guard_margin", 0.02)) - fingerprint_guard_margin) <= 1e-12
    and bool(config.get("train_connector", False)) == train_connector
)
sys.exit(0 if ok else 1)
PY
  then
    DIFFUSION_ARGS+=(--resume-checkpoint "$RESUME_DIFFUSION_CHECKPOINT")
  elif [ "$ALLOW_INCOMPATIBLE_RESUME_WEIGHTS" = "1" ]; then
    echo "Resume checkpoint predates current diffusion/source-guard settings; warm-starting model weights only."
    DIFFUSION_ARGS+=(--resume-checkpoint "$RESUME_DIFFUSION_CHECKPOINT")
  else
    echo "Resume checkpoint predates current diffusion/source-guard settings; retraining latent diffusion."
  fi
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_latent_diffusion_generation.py" "${DIFFUSION_ARGS[@]}"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_latent_diffusion_generation.py" \
  --eval-jsonl "$EVAL_JSONL" \
  --condition-connector "$CONNECTOR" \
  --diffusion-checkpoint "$DIFFUSION_DIR/latent_diffusion_generation.pt" \
  --output-dir "$EVAL_DIR" \
  --limit "$EVAL_LIMIT" \
  --max-eval-per-property-count "$MAX_EVAL_PER_PROPERTY_COUNT" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --sample-steps "$EVAL_SAMPLE_STEPS" \
  --sample-eta "$EVAL_SAMPLE_ETA" \
  --device "$DEVICE" \
  --seed "$EVAL_SEED"

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
