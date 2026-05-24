#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

pick_python() {
  local candidates=()
  if [[ -n "${SKETCHIMAGE_PYTHON_BIN:-}" ]]; then
    candidates+=("$SKETCHIMAGE_PYTHON_BIN")
  fi
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("$PYTHON_BIN")
  fi
  candidates+=("python3")
  candidates+=("/Users/dongpochen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")

  local candidate
  for candidate in "${candidates[@]}"; do
    if ! command -v "$candidate" >/dev/null 2>&1 && [[ ! -x "$candidate" ]]; then
      continue
    fi
    if "$candidate" - <<'PY' >/dev/null 2>&1
import numpy
PY
    then
      echo "$candidate"
      return 0
    fi
  done

  echo "No Python with numpy found. Set SKETCHIMAGE_PYTHON_BIN=/path/to/python3." >&2
  return 1
}

PYTHON_BIN="$(pick_python)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

PRESET="${SKETCHIMAGE_PRESET:-sketchmol_aligned}"
if [[ "$PRESET" == "sketchmol_aligned" ]]; then
  DEFAULT_FEATURE_DIM=256
  DEFAULT_LATENT_DIM=4096
  DEFAULT_TOP_K=8
  DEFAULT_TRAIN_FRACTION=0.8
  DEFAULT_RENDER_IMAGE_CONTEXT=1
else
  DEFAULT_FEATURE_DIM=96
  DEFAULT_LATENT_DIM=48
  DEFAULT_TOP_K=5
  DEFAULT_TRAIN_FRACTION=0.67
  DEFAULT_RENDER_IMAGE_CONTEXT=1
fi

RUN_NAME="${SKETCHIMAGE_RUN_NAME:-sketchimage_jepa_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${SKETCHIMAGE_RUN_ROOT:-outputs/runs/$RUN_NAME}"
MOLECULE_CSV="${SKETCHIMAGE_MOLECULE_CSV:-}"
DATASET_CSV="${SKETCHIMAGE_DATASET_CSV:-data/example_tasks.csv}"
TRAIN_CSV="${SKETCHIMAGE_TRAIN_CSV:-}"
EVAL_CSV="${SKETCHIMAGE_EVAL_CSV:-}"
FEATURE_DIM="${SKETCHIMAGE_FEATURE_DIM:-$DEFAULT_FEATURE_DIM}"
LATENT_DIM="${SKETCHIMAGE_LATENT_DIM:-$DEFAULT_LATENT_DIM}"
TOP_K="${SKETCHIMAGE_TOP_K:-$DEFAULT_TOP_K}"
RIDGE="${SKETCHIMAGE_RIDGE:-0.001}"
BACKEND="${SKETCHIMAGE_BACKEND:-ridge}"
TORCH_HIDDEN_DIM="${SKETCHIMAGE_TORCH_HIDDEN_DIM:-1024}"
TORCH_EPOCHS="${SKETCHIMAGE_TORCH_EPOCHS:-20}"
TORCH_BATCH_SIZE="${SKETCHIMAGE_TORCH_BATCH_SIZE:-128}"
TORCH_LR="${SKETCHIMAGE_TORCH_LR:-0.001}"
TORCH_WEIGHT_DECAY="${SKETCHIMAGE_TORCH_WEIGHT_DECAY:-0.0001}"
TORCH_DIFFUSION_STEPS="${SKETCHIMAGE_TORCH_DIFFUSION_STEPS:-16}"
TORCH_TRAIN_NOISE="${SKETCHIMAGE_TORCH_TRAIN_NOISE:-0.35}"
TORCH_DIRECT_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_DIRECT_LOSS_WEIGHT:-1.0}"
TORCH_COSINE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_COSINE_LOSS_WEIGHT:-1.0}"
TORCH_POSITIVE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_POSITIVE_LOSS_WEIGHT:-8.0}"
TORCH_CONTRASTIVE_LOSS_WEIGHT="${SKETCHIMAGE_TORCH_CONTRASTIVE_LOSS_WEIGHT:-0.25}"
TORCH_CONTRASTIVE_TEMPERATURE="${SKETCHIMAGE_TORCH_CONTRASTIVE_TEMPERATURE:-0.10}"
TORCH_DEVICE="${SKETCHIMAGE_TORCH_DEVICE:-auto}"
DE_NOVO_LATENT_RERANK_WEIGHT="${SKETCHIMAGE_DE_NOVO_LATENT_RERANK_WEIGHT:-0.05}"
SOURCE_RERANK_WEIGHT="${SKETCHIMAGE_SOURCE_RERANK_WEIGHT:-0.35}"
PROPERTY_RERANK_WEIGHT="${SKETCHIMAGE_PROPERTY_RERANK_WEIGHT:-0.25}"
SCAFFOLD_RERANK_BONUS="${SKETCHIMAGE_SCAFFOLD_RERANK_BONUS:-0.15}"
TRAIN_FRACTION="${SKETCHIMAGE_TRAIN_FRACTION:-$DEFAULT_TRAIN_FRACTION}"
SEED="${SKETCHIMAGE_SEED:-7}"
LIMIT="${SKETCHIMAGE_LIMIT:-}"
RUN_TESTS="${SKETCHIMAGE_RUN_TESTS:-1}"
RENDER_IMAGE_CONTEXT="${SKETCHIMAGE_RENDER_IMAGE_CONTEXT:-$DEFAULT_RENDER_IMAGE_CONTEXT}"

if [[ "$BACKEND" == "torch_denoiser" ]]; then
  "$PYTHON_BIN" - <<'PY'
try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: SKETCHIMAGE_BACKEND=torch_denoiser requires PyTorch in this Python: {exc}")
print("torch=", torch.__version__)
print("torch_cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch_cuda_device=", torch.cuda.get_device_name(0))
PY
fi

echo "SketchImage-JEPA one-click run"
echo "  python=$PYTHON_BIN"
echo "  run_root=$RUN_ROOT"
echo "  molecule_csv=${MOLECULE_CSV:-<not provided>}"
echo "  dataset_csv=$DATASET_CSV"
echo "  train_csv=${TRAIN_CSV:-<auto split>}"
echo "  eval_csv=${EVAL_CSV:-<auto split>}"
echo "  feature_dim=$FEATURE_DIM"
echo "  latent_dim=$LATENT_DIM"
echo "  top_k=$TOP_K"
echo "  ridge=$RIDGE"
echo "  backend=$BACKEND"
if [[ "$BACKEND" == "torch_denoiser" ]]; then
  echo "  torch_hidden_dim=$TORCH_HIDDEN_DIM"
  echo "  torch_epochs=$TORCH_EPOCHS"
  echo "  torch_batch_size=$TORCH_BATCH_SIZE"
  echo "  torch_lr=$TORCH_LR"
  echo "  torch_diffusion_steps=$TORCH_DIFFUSION_STEPS"
  echo "  torch_direct_loss_weight=$TORCH_DIRECT_LOSS_WEIGHT"
  echo "  torch_cosine_loss_weight=$TORCH_COSINE_LOSS_WEIGHT"
  echo "  torch_positive_loss_weight=$TORCH_POSITIVE_LOSS_WEIGHT"
  echo "  torch_contrastive_loss_weight=$TORCH_CONTRASTIVE_LOSS_WEIGHT"
  echo "  torch_contrastive_temperature=$TORCH_CONTRASTIVE_TEMPERATURE"
  echo "  torch_device=$TORCH_DEVICE"
fi
echo "  de_novo_latent_rerank_weight=$DE_NOVO_LATENT_RERANK_WEIGHT"
echo "  source_rerank_weight=$SOURCE_RERANK_WEIGHT"
echo "  property_rerank_weight=$PROPERTY_RERANK_WEIGHT"
echo "  scaffold_rerank_bonus=$SCAFFOLD_RERANK_BONUS"
echo "  train_fraction=$TRAIN_FRACTION"
echo "  seed=$SEED"
echo "  render_image_context=$RENDER_IMAGE_CONTEXT"

check_input_file() {
  local label="$1"
  local path="$2"
  if [[ -n "$path" && ! -f "$path" ]]; then
    echo "ERROR: $label does not exist: $path" >&2
    echo "Run from $ROOT_DIR, or provide an absolute CSV path." >&2
    exit 2
  fi
}

check_input_file "SKETCHIMAGE_MOLECULE_CSV" "$MOLECULE_CSV"
check_input_file "SKETCHIMAGE_DATASET_CSV" "$DATASET_CSV"
check_input_file "SKETCHIMAGE_TRAIN_CSV" "$TRAIN_CSV"
check_input_file "SKETCHIMAGE_EVAL_CSV" "$EVAL_CSV"

if [[ "$RUN_TESTS" == "1" ]]; then
  echo
  echo "[1/2] Running tests"
  "$PYTHON_BIN" -m unittest discover -s tests
else
  echo
  echo "[1/2] Skipping tests because SKETCHIMAGE_RUN_TESTS=$RUN_TESTS"
fi

echo
echo "[2/2] Preparing data and running experiment"

if [[ -n "$MOLECULE_CSV" && -z "$TRAIN_CSV" && -z "$EVAL_CSV" ]]; then
  TASK_CSV="${SKETCHIMAGE_TASK_CSV:-outputs/tasks/${RUN_NAME}_tasks.csv}"
  echo "  building task CSV from molecule CSV"
  echo "  task_csv=$TASK_CSV"
  "$PYTHON_BIN" -m sketchimage_jepa.task_builder \
    --molecule-csv "$MOLECULE_CSV" \
    --out "$TASK_CSV" \
    --limit "${SKETCHIMAGE_MOLECULE_LIMIT:-10000}" \
    --max-tasks "${SKETCHIMAGE_MAX_TASKS:-5000}" \
    --pairs-per-source "${SKETCHIMAGE_PAIRS_PER_SOURCE:-2}" \
    --pair-candidates "${SKETCHIMAGE_PAIR_CANDIDATES:-128}" \
    --min-similarity "${SKETCHIMAGE_MIN_SIMILARITY:-0.15}" \
    --max-similarity "${SKETCHIMAGE_MAX_SIMILARITY:-0.90}" \
    --seed "$SEED" \
    --task-types "${SKETCHIMAGE_TASK_TYPES:-de_novo,edit,inpaint,fragment_grow}"
  DATASET_CSV="$TASK_CSV"
fi

ARGS=(
  -m sketchimage_jepa.experiment
  --output-dir "$RUN_ROOT"
  --feature-dim "$FEATURE_DIM"
  --latent-dim "$LATENT_DIM"
  --top-k "$TOP_K"
  --ridge "$RIDGE"
  --backend "$BACKEND"
  --seed "$SEED"
  --preset "$PRESET"
)

if [[ "$BACKEND" == "torch_denoiser" ]]; then
  ARGS+=(
    --torch-hidden-dim "$TORCH_HIDDEN_DIM"
    --torch-epochs "$TORCH_EPOCHS"
    --torch-batch-size "$TORCH_BATCH_SIZE"
    --torch-lr "$TORCH_LR"
    --torch-weight-decay "$TORCH_WEIGHT_DECAY"
    --torch-diffusion-steps "$TORCH_DIFFUSION_STEPS"
    --torch-train-noise "$TORCH_TRAIN_NOISE"
    --torch-direct-loss-weight "$TORCH_DIRECT_LOSS_WEIGHT"
    --torch-cosine-loss-weight "$TORCH_COSINE_LOSS_WEIGHT"
    --torch-positive-loss-weight "$TORCH_POSITIVE_LOSS_WEIGHT"
    --torch-contrastive-loss-weight "$TORCH_CONTRASTIVE_LOSS_WEIGHT"
    --torch-contrastive-temperature "$TORCH_CONTRASTIVE_TEMPERATURE"
    --torch-device "$TORCH_DEVICE"
  )
fi

ARGS+=(--de-novo-latent-rerank-weight "$DE_NOVO_LATENT_RERANK_WEIGHT")
ARGS+=(--source-rerank-weight "$SOURCE_RERANK_WEIGHT")
ARGS+=(--property-rerank-weight "$PROPERTY_RERANK_WEIGHT")
ARGS+=(--scaffold-rerank-bonus "$SCAFFOLD_RERANK_BONUS")

if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

if [[ -n "$TRAIN_CSV" || -n "$EVAL_CSV" ]]; then
  if [[ -z "$TRAIN_CSV" || -z "$EVAL_CSV" ]]; then
    echo "Provide both SKETCHIMAGE_TRAIN_CSV and SKETCHIMAGE_EVAL_CSV, or neither." >&2
    exit 2
  fi
  ARGS+=(--train-csv "$TRAIN_CSV" --eval-csv "$EVAL_CSV")
else
  ARGS+=(--dataset-csv "$DATASET_CSV" --train-fraction "$TRAIN_FRACTION")
fi

if [[ "$RENDER_IMAGE_CONTEXT" == "1" ]]; then
  ARGS+=(--render-image-context)
fi

"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "SketchImage-JEPA run finished: $RUN_ROOT"
echo "  metrics=$RUN_ROOT/metrics.json"
echo "  predictions=$RUN_ROOT/predictions.csv"
echo "  config=$RUN_ROOT/run_config.json"
