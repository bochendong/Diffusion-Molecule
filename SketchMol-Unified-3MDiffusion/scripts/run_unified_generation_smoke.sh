#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="SketchMol-Unified-3MDiffusion"
REPO_DIR="$(pwd)"
PYTHON_BIN="${SMU3M_PYTHON_BIN:-python3}"
THREE_M_ROOT="${SMU3M_3M_ROOT:-Research/Molecule Generation/3M-Diffusion}"
THREE_M_GIT_URL="${SMU3M_3M_GIT_URL:-https://github.com/huaishengzhu/3MDiffusion}"
AUTO_CLONE_3M="${SMU3M_AUTO_CLONE_3M:-1}"
EDIT_MANIFEST="${SMU3M_EDIT_MANIFEST:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/diffusion_edit_manifest.csv}"
OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_smoke}"

DESCRIPTION_LIMIT="${SMU3M_DESCRIPTION_LIMIT:-200}"
EDIT_LIMIT="${SMU3M_EDIT_LIMIT:-500}"
TRAIN_LIMIT="${SMU3M_TRAIN_LIMIT:-500}"
BATCH_SIZE="${SMU3M_BATCH_SIZE:-32}"
NUM_WORKERS="${SMU3M_NUM_WORKERS:-0}"
PIN_MEMORY="${SMU3M_PIN_MEMORY:-0}"
EPOCHS="${SMU3M_EPOCHS:-1}"
EVAL_LIMIT="${SMU3M_EVAL_LIMIT:-1000}"
EVAL_BATCH_SIZE="${SMU3M_EVAL_BATCH_SIZE:-64}"
EVAL_SAMPLE_STEPS="${SMU3M_EVAL_SAMPLE_STEPS:-20}"
EVAL_SAMPLE_ETA="${SMU3M_EVAL_SAMPLE_ETA:-0.0}"
DIFFUSION_TIMESTEPS="${SMU3M_DIFFUSION_TIMESTEPS:-100}"
DIFFUSION_OBJECTIVE="${SMU3M_DIFFUSION_OBJECTIVE:-pred_x0}"
DIFFUSION_TARGET="${SMU3M_DIFFUSION_TARGET:-residual}"
PRIOR_LOSS_WEIGHT="${SMU3M_PRIOR_LOSS_WEIGHT:-0.0}"
ALIGNMENT_HIDDEN_DIM="${SMU3M_ALIGNMENT_HIDDEN_DIM:-512}"
EDIT_HIDDEN_DIM="${SMU3M_EDIT_HIDDEN_DIM:-512}"
DIFFUSION_HIDDEN_DIM="${SMU3M_DIFFUSION_HIDDEN_DIM:-512}"
NUM_QUERIES="${SMU3M_NUM_QUERIES:-16}"
DIFFUSION_DEPTH="${SMU3M_DIFFUSION_DEPTH:-4}"
DEVICE="${SMU3M_DEVICE:-auto}"
CHECKPOINT_EVERY="${SMU3M_CHECKPOINT_EVERY:-1}"
RESUME="${SMU3M_RESUME:-1}"
REQUIRE_CUDA="${SMU3M_REQUIRE_CUDA:-0}"
INCLUDE_PUBCHEM="${SMU3M_INCLUDE_PUBCHEM:-0}"
INCLUDE_KV="${SMU3M_INCLUDE_KV:-0}"

echo "Running unified 3M Understanding + latent diffusion pipeline"
echo "  python=$PYTHON_BIN"
echo "  3m_root=$THREE_M_ROOT"
echo "  edit_manifest=$EDIT_MANIFEST"
echo "  output_dir=$OUTPUT_DIR"
echo "  device=$DEVICE"
echo "  require_cuda=$REQUIRE_CUDA"
echo "  diffusion_timesteps=$DIFFUSION_TIMESTEPS"
echo "  diffusion_objective=$DIFFUSION_OBJECTIVE"
echo "  diffusion_target=$DIFFUSION_TARGET"
echo "  eval_sample_eta=$EVAL_SAMPLE_ETA"
echo "  batch_size=$BATCH_SIZE"
echo "  num_workers=$NUM_WORKERS"
echo "  pin_memory=$PIN_MEMORY"
echo "  alignment_hidden_dim=$ALIGNMENT_HIDDEN_DIM"
echo "  edit_hidden_dim=$EDIT_HIDDEN_DIM"
echo "  diffusion_hidden_dim=$DIFFUSION_HIDDEN_DIM"
echo "  num_queries=$NUM_QUERIES"
echo "  diffusion_depth=$DIFFUSION_DEPTH"
echo "  include_pubchem=$INCLUDE_PUBCHEM"
echo "  include_kv=$INCLUDE_KV"

if [ ! -d "$THREE_M_ROOT/data" ]; then
  if [ "$AUTO_CLONE_3M" = "1" ] && command -v git >/dev/null 2>&1; then
    echo "3M-Diffusion data not found; cloning reference repo:"
    echo "  url=$THREE_M_GIT_URL"
    echo "  dest=$THREE_M_ROOT"
    mkdir -p "$(dirname "$THREE_M_ROOT")"
    git clone "$THREE_M_GIT_URL" "$THREE_M_ROOT"
  else
    echo "Missing 3M-Diffusion data directory: $THREE_M_ROOT/data" >&2
    echo "Set SMU3M_3M_ROOT or enable SMU3M_AUTO_CLONE_3M=1." >&2
    exit 2
  fi
fi

if [ ! -f "$EDIT_MANIFEST" ]; then
  echo "Missing edit manifest: $EDIT_MANIFEST" >&2
  echo "Build the multiproperty dataset first, then rerun this script." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

PREFLIGHT_ARGS=(
  --three-m-root "$THREE_M_ROOT"
  --edit-manifest "$EDIT_MANIFEST"
)
if [ "$REQUIRE_CUDA" = "1" ]; then
  PREFLIGHT_ARGS+=(--require-cuda)
fi
if [ "$INCLUDE_PUBCHEM" = "1" ]; then
  PREFLIGHT_ARGS+=(--include-pubchem)
fi
if [ "$INCLUDE_KV" = "1" ]; then
  PREFLIGHT_ARGS+=(--include-kv)
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/preflight_unified_3m.py" "${PREFLIGHT_ARGS[@]}"

EXPORT_ARGS=(
  --three-m-root "$THREE_M_ROOT" \
  --edit-manifest "$EDIT_MANIFEST" \
  --description-limit-per-split "$DESCRIPTION_LIMIT" \
  --edit-limit "$EDIT_LIMIT" \
  --output-dir "$OUTPUT_DIR/dataset"
)
if [ "$INCLUDE_PUBCHEM" = "1" ]; then
  EXPORT_ARGS+=(--include-pubchem)
fi
if [ "$INCLUDE_KV" = "1" ]; then
  EXPORT_ARGS+=(--include-kv)
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/export_unified_condition_dataset.py" "${EXPORT_ARGS[@]}"

ALIGNMENT_ARGS=(
  --train-jsonl "$OUTPUT_DIR/dataset/unified_condition_train.jsonl" \
  --output-dir "$OUTPUT_DIR/alignment" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --limit "$TRAIN_LIMIT" \
  --hidden-dim "$ALIGNMENT_HIDDEN_DIM" \
  --device "$DEVICE" \
  --checkpoint-every "$CHECKPOINT_EVERY"
)
if [ "$PIN_MEMORY" = "1" ]; then
  ALIGNMENT_ARGS+=(--pin-memory)
fi
if [ "$RESUME" = "1" ] && [ -f "$OUTPUT_DIR/alignment/checkpoints/latest.pt" ]; then
  ALIGNMENT_ARGS+=(--resume-checkpoint "$OUTPUT_DIR/alignment/checkpoints/latest.pt")
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_alignment_pretraining.py" "${ALIGNMENT_ARGS[@]}"

EDIT_ARGS=(
  --train-jsonl "$OUTPUT_DIR/dataset/unified_condition_train.jsonl" \
  --output-dir "$OUTPUT_DIR/edit_condition_tokens" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --limit "$TRAIN_LIMIT" \
  --hidden-dim "$EDIT_HIDDEN_DIM" \
  --num-queries "$NUM_QUERIES" \
  --device "$DEVICE" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --export-features
)
if [ "$PIN_MEMORY" = "1" ]; then
  EDIT_ARGS+=(--pin-memory)
fi
if [ "$RESUME" = "1" ] && [ -f "$OUTPUT_DIR/edit_condition_tokens/checkpoints/latest.pt" ]; then
  EDIT_ARGS+=(--resume-checkpoint "$OUTPUT_DIR/edit_condition_tokens/checkpoints/latest.pt")
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_edit_condition_tokens.py" "${EDIT_ARGS[@]}"

DIFFUSION_ARGS=(
  --train-jsonl "$OUTPUT_DIR/dataset/unified_condition_train.jsonl" \
  --condition-connector "$OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt" \
  --output-dir "$OUTPUT_DIR/latent_diffusion" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --epochs "$EPOCHS" \
  --limit "$TRAIN_LIMIT" \
  --timesteps "$DIFFUSION_TIMESTEPS" \
  --diffusion-objective "$DIFFUSION_OBJECTIVE" \
  --diffusion-target "$DIFFUSION_TARGET" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --hidden-dim "$DIFFUSION_HIDDEN_DIM" \
  --depth "$DIFFUSION_DEPTH" \
  --device "$DEVICE" \
  --checkpoint-every "$CHECKPOINT_EVERY"
)
if [ "$PIN_MEMORY" = "1" ]; then
  DIFFUSION_ARGS+=(--pin-memory)
fi
if [ "$RESUME" = "1" ] && [ -f "$OUTPUT_DIR/latent_diffusion/checkpoints/latest.pt" ]; then
  if [ -f "$OUTPUT_DIR/latent_diffusion/metrics.json" ] \
    && grep -q "\"diffusion_objective\": \"$DIFFUSION_OBJECTIVE\"" "$OUTPUT_DIR/latent_diffusion/metrics.json" \
    && grep -q "\"diffusion_target\": \"$DIFFUSION_TARGET\"" "$OUTPUT_DIR/latent_diffusion/metrics.json"; then
    DIFFUSION_ARGS+=(--resume-checkpoint "$OUTPUT_DIR/latent_diffusion/checkpoints/latest.pt")
  else
    echo "Existing latent diffusion checkpoint predates $DIFFUSION_OBJECTIVE/$DIFFUSION_TARGET settings; retraining latent diffusion."
  fi
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_latent_diffusion_generation.py" "${DIFFUSION_ARGS[@]}"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_latent_diffusion_generation.py" \
  --eval-jsonl "$OUTPUT_DIR/dataset/unified_condition_eval.jsonl" \
  --condition-connector "$OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt" \
  --diffusion-checkpoint "$OUTPUT_DIR/latent_diffusion/latent_diffusion_generation.pt" \
  --output-dir "$OUTPUT_DIR/eval_latent" \
  --limit "$EVAL_LIMIT" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --sample-steps "$EVAL_SAMPLE_STEPS" \
  --sample-eta "$EVAL_SAMPLE_ETA" \
  --device "$DEVICE"

echo "Unified smoke finished:"
echo "  dataset=$OUTPUT_DIR/dataset/summary.json"
echo "  alignment=$OUTPUT_DIR/alignment/alignment_model.pt"
echo "  connector=$OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt"
echo "  diffusion=$OUTPUT_DIR/latent_diffusion/latent_diffusion_generation.pt"
echo "  eval=$OUTPUT_DIR/eval_latent/metrics.json"
