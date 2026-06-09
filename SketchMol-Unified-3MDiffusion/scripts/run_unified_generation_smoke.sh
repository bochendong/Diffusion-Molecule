#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="SketchMol-Unified-3MDiffusion"
REPO_DIR="$(pwd)"
PYTHON_BIN="${SMU3M_PYTHON_BIN:-python3}"
THREE_M_ROOT="${SMU3M_3M_ROOT:-Research/Molecule Generation/3M-Diffusion}"
THREE_M_GIT_URL="${SMU3M_3M_GIT_URL:-https://github.com/huaishengzhu/3MDiffusion}"
AUTO_CLONE_3M="${SMU3M_AUTO_CLONE_3M:-1}"
EDIT_MANIFEST="${SMU3M_EDIT_MANIFEST:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_source_neighbor_v1/diffusion_edit_manifest.csv}"
DATASET_MODE="${SMU3M_DATASET_MODE:-multiproperty}"
DM_DATA_ROOT="${DM_DATA_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule}"
MOLEDIT_TRAIN_SPLIT="${SMU3M_MOLEDIT_TRAIN_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/train.csv}"
MOLEDIT_EVAL_SPLIT="${SMU3M_MOLEDIT_EVAL_SPLIT:-$DM_DATA_ROOT/processed/moledit-instruct/enhanced_v1/splits/eval_balanced.csv}"
OUTPUT_DIR="${SMU3M_OUTPUT_DIR:-SketchMol-Unified-3MDiffusion/outputs/unified_generation_smoke}"

DESCRIPTION_LIMIT="${SMU3M_DESCRIPTION_LIMIT:-200}"
EDIT_LIMIT="${SMU3M_EDIT_LIMIT:-500}"
MOLEDIT_TRAIN_LIMIT="${SMU3M_MOLEDIT_TRAIN_LIMIT:-$EDIT_LIMIT}"
MOLEDIT_EVAL_LIMIT="${SMU3M_MOLEDIT_EVAL_LIMIT:-}"
TRAIN_LIMIT="${SMU3M_TRAIN_LIMIT:-500}"
BATCH_SIZE="${SMU3M_BATCH_SIZE:-32}"
NUM_WORKERS="${SMU3M_NUM_WORKERS:-0}"
PIN_MEMORY="${SMU3M_PIN_MEMORY:-0}"
EPOCHS="${SMU3M_EPOCHS:-1}"
EVAL_LIMIT="${SMU3M_EVAL_LIMIT:-1000}"
MAX_EVAL_PER_PROPERTY_COUNT="${SMU3M_MAX_EVAL_PER_PROPERTY_COUNT:-}"
EVAL_BATCH_SIZE="${SMU3M_EVAL_BATCH_SIZE:-64}"
EVAL_SAMPLE_STEPS="${SMU3M_EVAL_SAMPLE_STEPS:-20}"
EVAL_SAMPLE_ETA="${SMU3M_EVAL_SAMPLE_ETA:-0.0}"
ALIGNMENT_SEED="${SMU3M_ALIGNMENT_SEED:-7}"
EDIT_SEED="${SMU3M_EDIT_SEED:-11}"
DIFFUSION_SEED="${SMU3M_DIFFUSION_SEED:-13}"
EVAL_SEED="${SMU3M_EVAL_SEED:-17}"
DIFFUSION_TIMESTEPS="${SMU3M_DIFFUSION_TIMESTEPS:-100}"
DIFFUSION_OBJECTIVE="${SMU3M_DIFFUSION_OBJECTIVE:-pred_x0}"
DIFFUSION_TARGET="${SMU3M_DIFFUSION_TARGET:-residual}"
PRIOR_LOSS_WEIGHT="${SMU3M_PRIOR_LOSS_WEIGHT:-0.05}"
SOURCE_REGRET_LOSS_WEIGHT="${SMU3M_SOURCE_REGRET_LOSS_WEIGHT:-0.35}"
SOURCE_REGRET_MARGIN="${SMU3M_SOURCE_REGRET_MARGIN:-0.0}"
SOURCE_RADIUS_LOSS_WEIGHT="${SMU3M_SOURCE_RADIUS_LOSS_WEIGHT:-0.10}"
SOURCE_RADIUS_MARGIN="${SMU3M_SOURCE_RADIUS_MARGIN:-0.05}"
SOURCE_SIMILARITY_WEIGHT_FLOOR="${SMU3M_SOURCE_SIMILARITY_WEIGHT_FLOOR:-0.25}"
SOURCE_FINGERPRINT_PRIOR_BLEND="${SMU3M_SOURCE_FINGERPRINT_PRIOR_BLEND:-0.85}"
FINGERPRINT_GUARD_LOSS_WEIGHT="${SMU3M_FINGERPRINT_GUARD_LOSS_WEIGHT:-0.50}"
FINGERPRINT_GUARD_MARGIN="${SMU3M_FINGERPRINT_GUARD_MARGIN:-0.02}"
DIFFUSION_EPOCHS="${SMU3M_DIFFUSION_EPOCHS:-$EPOCHS}"
DIFFUSION_LR="${SMU3M_DIFFUSION_LR:-1e-3}"
TRAIN_DIFFUSION_CONNECTOR="${SMU3M_TRAIN_DIFFUSION_CONNECTOR:-1}"
ALIGNMENT_HIDDEN_DIM="${SMU3M_ALIGNMENT_HIDDEN_DIM:-512}"
EDIT_HIDDEN_DIM="${SMU3M_EDIT_HIDDEN_DIM:-512}"
DIFFUSION_HIDDEN_DIM="${SMU3M_DIFFUSION_HIDDEN_DIM:-512}"
NUM_QUERIES="${SMU3M_NUM_QUERIES:-16}"
DIFFUSION_DEPTH="${SMU3M_DIFFUSION_DEPTH:-4}"
SOURCE_SIMILARITY_LOSS_WEIGHT="${SMU3M_SOURCE_SIMILARITY_LOSS_WEIGHT:-0.15}"
HARD_NEGATIVE_LOSS_WEIGHT="${SMU3M_HARD_NEGATIVE_LOSS_WEIGHT:-0.05}"
SOURCE_AWARE_TEMPERATURE="${SMU3M_SOURCE_AWARE_TEMPERATURE:-0.07}"
HARD_NEGATIVE_MARGIN="${SMU3M_HARD_NEGATIVE_MARGIN:-0.2}"
SOURCE_AWARE_SHARED_GRADIENT="${SMU3M_SOURCE_AWARE_SHARED_GRADIENT:-0}"
DEVICE="${SMU3M_DEVICE:-auto}"
CHECKPOINT_EVERY="${SMU3M_CHECKPOINT_EVERY:-1}"
RESUME="${SMU3M_RESUME:-1}"
REQUIRE_CUDA="${SMU3M_REQUIRE_CUDA:-0}"
INCLUDE_PUBCHEM="${SMU3M_INCLUDE_PUBCHEM:-0}"
INCLUDE_KV="${SMU3M_INCLUDE_KV:-0}"
if [ "$DATASET_MODE" = "moledit" ]; then
  MIN_EDIT_SOURCE_TANIMOTO="${SMU3M_MIN_EDIT_SOURCE_TANIMOTO:-0.0}"
  REQUIRE_EDIT_QUALITY_COLUMNS="${SMU3M_REQUIRE_EDIT_QUALITY_COLUMNS:-0}"
  REQUIRE_EVAL_ORACLE_STRICT="${SMU3M_REQUIRE_EVAL_ORACLE_STRICT:-0}"
else
  MIN_EDIT_SOURCE_TANIMOTO="${SMU3M_MIN_EDIT_SOURCE_TANIMOTO:-0.4}"
  REQUIRE_EDIT_QUALITY_COLUMNS="${SMU3M_REQUIRE_EDIT_QUALITY_COLUMNS:-1}"
  REQUIRE_EVAL_ORACLE_STRICT="${SMU3M_REQUIRE_EVAL_ORACLE_STRICT:-1}"
fi
MOLEDIT_TABLE1_TASKS_ONLY="${SMU3M_MOLEDIT_TABLE1_TASKS_ONLY:-0}"

echo "Running unified 3M Understanding + latent diffusion pipeline"
echo "  python=$PYTHON_BIN"
echo "  3m_root=$THREE_M_ROOT"
echo "  dataset_mode=$DATASET_MODE"
echo "  edit_manifest=$EDIT_MANIFEST"
echo "  moledit_train_split=$MOLEDIT_TRAIN_SPLIT"
echo "  moledit_eval_split=$MOLEDIT_EVAL_SPLIT"
echo "  output_dir=$OUTPUT_DIR"
echo "  device=$DEVICE"
echo "  require_cuda=$REQUIRE_CUDA"
echo "  diffusion_timesteps=$DIFFUSION_TIMESTEPS"
echo "  diffusion_objective=$DIFFUSION_OBJECTIVE"
echo "  diffusion_target=$DIFFUSION_TARGET"
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
echo "  eval_limit=$EVAL_LIMIT"
echo "  max_eval_per_property_count=${MAX_EVAL_PER_PROPERTY_COUNT:-none}"
echo "  eval_sample_steps=$EVAL_SAMPLE_STEPS"
echo "  eval_sample_eta=$EVAL_SAMPLE_ETA"
echo "  alignment_seed=$ALIGNMENT_SEED"
echo "  edit_seed=$EDIT_SEED"
echo "  diffusion_seed=$DIFFUSION_SEED"
echo "  eval_seed=$EVAL_SEED"
echo "  batch_size=$BATCH_SIZE"
echo "  num_workers=$NUM_WORKERS"
echo "  pin_memory=$PIN_MEMORY"
echo "  alignment_hidden_dim=$ALIGNMENT_HIDDEN_DIM"
echo "  edit_hidden_dim=$EDIT_HIDDEN_DIM"
echo "  diffusion_hidden_dim=$DIFFUSION_HIDDEN_DIM"
echo "  num_queries=$NUM_QUERIES"
echo "  diffusion_depth=$DIFFUSION_DEPTH"
echo "  source_similarity_loss_weight=$SOURCE_SIMILARITY_LOSS_WEIGHT"
echo "  hard_negative_loss_weight=$HARD_NEGATIVE_LOSS_WEIGHT"
echo "  source_aware_temperature=$SOURCE_AWARE_TEMPERATURE"
echo "  hard_negative_margin=$HARD_NEGATIVE_MARGIN"
echo "  source_aware_shared_gradient=$SOURCE_AWARE_SHARED_GRADIENT"
echo "  include_pubchem=$INCLUDE_PUBCHEM"
echo "  include_kv=$INCLUDE_KV"
echo "  min_edit_source_tanimoto=$MIN_EDIT_SOURCE_TANIMOTO"
echo "  require_edit_quality_columns=$REQUIRE_EDIT_QUALITY_COLUMNS"
echo "  require_eval_oracle_strict=$REQUIRE_EVAL_ORACLE_STRICT"
echo "  edit_limit=$EDIT_LIMIT"
echo "  moledit_train_limit=${MOLEDIT_TRAIN_LIMIT:-none}"
echo "  moledit_eval_limit=${MOLEDIT_EVAL_LIMIT:-none}"
echo "  moledit_table1_tasks_only=$MOLEDIT_TABLE1_TASKS_ONLY"

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

if [ "$DATASET_MODE" = "moledit" ]; then
  if [ ! -f "$MOLEDIT_TRAIN_SPLIT" ]; then
    echo "Missing MolEdit train split: $MOLEDIT_TRAIN_SPLIT" >&2
    exit 2
  fi
  if [ ! -f "$MOLEDIT_EVAL_SPLIT" ]; then
    echo "Missing MolEdit eval split: $MOLEDIT_EVAL_SPLIT" >&2
    exit 2
  fi
elif [ ! -f "$EDIT_MANIFEST" ]; then
  echo "Missing edit manifest: $EDIT_MANIFEST" >&2
  echo "Build the multiproperty dataset first, then rerun this script." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

PREFLIGHT_ARGS=(
  --three-m-root "$THREE_M_ROOT"
  --min-edit-source-tanimoto "$MIN_EDIT_SOURCE_TANIMOTO"
)
if [ "$DATASET_MODE" = "moledit" ]; then
  PREFLIGHT_ARGS+=(--moledit-train-split "$MOLEDIT_TRAIN_SPLIT" --moledit-eval-split "$MOLEDIT_EVAL_SPLIT")
else
  PREFLIGHT_ARGS+=(--edit-manifest "$EDIT_MANIFEST")
fi
if [ "$REQUIRE_CUDA" = "1" ]; then
  PREFLIGHT_ARGS+=(--require-cuda)
fi
if [ "$REQUIRE_EDIT_QUALITY_COLUMNS" = "1" ]; then
  PREFLIGHT_ARGS+=(--require-edit-quality-columns)
fi
if [ "$REQUIRE_EVAL_ORACLE_STRICT" = "1" ]; then
  PREFLIGHT_ARGS+=(--require-eval-oracle-strict)
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
  --description-limit-per-split "$DESCRIPTION_LIMIT" \
  --min-edit-source-tanimoto "$MIN_EDIT_SOURCE_TANIMOTO" \
  --output-dir "$OUTPUT_DIR/dataset"
)
if [ "$DATASET_MODE" = "moledit" ]; then
  EXPORT_ARGS+=(--moledit-train-split "$MOLEDIT_TRAIN_SPLIT" --moledit-eval-split "$MOLEDIT_EVAL_SPLIT")
  if [[ -n "$MOLEDIT_TRAIN_LIMIT" ]]; then
    EXPORT_ARGS+=(--moledit-train-limit "$MOLEDIT_TRAIN_LIMIT")
  fi
  if [[ -n "$MOLEDIT_EVAL_LIMIT" ]]; then
    EXPORT_ARGS+=(--moledit-eval-limit "$MOLEDIT_EVAL_LIMIT")
  fi
  if [ "$MOLEDIT_TABLE1_TASKS_ONLY" = "1" ]; then
    EXPORT_ARGS+=(--moledit-table1-tasks-only)
  fi
else
  EXPORT_ARGS+=(--edit-manifest "$EDIT_MANIFEST" --edit-limit "$EDIT_LIMIT")
fi
if [ "$REQUIRE_EDIT_QUALITY_COLUMNS" = "1" ]; then
  EXPORT_ARGS+=(--require-edit-quality-columns)
fi
if [ "$REQUIRE_EVAL_ORACLE_STRICT" = "1" ]; then
  EXPORT_ARGS+=(--require-eval-oracle-strict)
fi
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
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --seed "$ALIGNMENT_SEED"
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
  --source-similarity-loss-weight "$SOURCE_SIMILARITY_LOSS_WEIGHT" \
  --hard-negative-loss-weight "$HARD_NEGATIVE_LOSS_WEIGHT" \
  --source-aware-temperature "$SOURCE_AWARE_TEMPERATURE" \
  --hard-negative-margin "$HARD_NEGATIVE_MARGIN" \
  --device "$DEVICE" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --seed "$EDIT_SEED" \
  --export-features
)
if [ "$SOURCE_AWARE_SHARED_GRADIENT" = "1" ]; then
  EDIT_ARGS+=(--source-aware-shared-gradient)
fi
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
  --epochs "$DIFFUSION_EPOCHS" \
  --limit "$TRAIN_LIMIT" \
  --timesteps "$DIFFUSION_TIMESTEPS" \
  --diffusion-objective "$DIFFUSION_OBJECTIVE" \
  --diffusion-target "$DIFFUSION_TARGET" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --source-regret-loss-weight "$SOURCE_REGRET_LOSS_WEIGHT" \
  --source-regret-margin "$SOURCE_REGRET_MARGIN" \
  --source-radius-loss-weight "$SOURCE_RADIUS_LOSS_WEIGHT" \
  --source-radius-margin "$SOURCE_RADIUS_MARGIN" \
  --source-similarity-weight-floor "$SOURCE_SIMILARITY_WEIGHT_FLOOR" \
  --source-fingerprint-prior-blend "$SOURCE_FINGERPRINT_PRIOR_BLEND" \
  --fingerprint-guard-loss-weight "$FINGERPRINT_GUARD_LOSS_WEIGHT" \
  --fingerprint-guard-margin "$FINGERPRINT_GUARD_MARGIN" \
  --lr "$DIFFUSION_LR" \
  --hidden-dim "$DIFFUSION_HIDDEN_DIM" \
  --depth "$DIFFUSION_DEPTH" \
  --device "$DEVICE" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --seed "$DIFFUSION_SEED"
)
if [ "$TRAIN_DIFFUSION_CONNECTOR" = "1" ]; then
  DIFFUSION_ARGS+=(--train-connector)
fi
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

EVAL_ARGS=(
  --eval-jsonl "$OUTPUT_DIR/dataset/unified_condition_eval.jsonl" \
  --condition-connector "$OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt" \
  --diffusion-checkpoint "$OUTPUT_DIR/latent_diffusion/latent_diffusion_generation.pt" \
  --output-dir "$OUTPUT_DIR/eval_latent" \
  --limit "$EVAL_LIMIT" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --sample-steps "$EVAL_SAMPLE_STEPS" \
  --sample-eta "$EVAL_SAMPLE_ETA" \
  --device "$DEVICE" \
  --seed "$EVAL_SEED"
)
if [[ -n "$MAX_EVAL_PER_PROPERTY_COUNT" ]]; then
  EVAL_ARGS+=(--max-eval-per-property-count "$MAX_EVAL_PER_PROPERTY_COUNT")
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_latent_diffusion_generation.py" "${EVAL_ARGS[@]}"

echo "Unified smoke finished:"
echo "  dataset=$OUTPUT_DIR/dataset/summary.json"
echo "  alignment=$OUTPUT_DIR/alignment/alignment_model.pt"
echo "  connector=$OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt"
echo "  diffusion=$OUTPUT_DIR/latent_diffusion/latent_diffusion_generation.pt"
echo "  eval=$OUTPUT_DIR/eval_latent/metrics.json"
