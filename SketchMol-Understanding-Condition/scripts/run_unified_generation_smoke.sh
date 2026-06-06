#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="SketchMol-Understanding-Condition"
REPO_DIR="$(pwd)"
PYTHON_BIN="${SUCC_PYTHON_BIN:-python3}"
THREE_M_ROOT="${SUCC_3M_ROOT:-Research/Molecule Generation/3M-Diffusion}"
THREE_M_GIT_URL="${SUCC_3M_GIT_URL:-https://github.com/huaishengzhu/3MDiffusion}"
AUTO_CLONE_3M="${SUCC_AUTO_CLONE_3M:-1}"
EDIT_MANIFEST="${SUCC_EDIT_MANIFEST:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1/diffusion_edit_manifest.csv}"
OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/unified_generation_smoke}"

DESCRIPTION_LIMIT="${SUCC_DESCRIPTION_LIMIT:-200}"
EDIT_LIMIT="${SUCC_EDIT_LIMIT:-500}"
TRAIN_LIMIT="${SUCC_TRAIN_LIMIT:-500}"
BATCH_SIZE="${SUCC_BATCH_SIZE:-32}"
EPOCHS="${SUCC_EPOCHS:-1}"
EVAL_LIMIT="${SUCC_EVAL_LIMIT:-1000}"
EVAL_BATCH_SIZE="${SUCC_EVAL_BATCH_SIZE:-64}"
EVAL_SAMPLE_STEPS="${SUCC_EVAL_SAMPLE_STEPS:-20}"

echo "Running unified Understanding + latent diffusion smoke"
echo "  python=$PYTHON_BIN"
echo "  3m_root=$THREE_M_ROOT"
echo "  edit_manifest=$EDIT_MANIFEST"
echo "  output_dir=$OUTPUT_DIR"

if [ ! -d "$THREE_M_ROOT/data" ]; then
  if [ "$AUTO_CLONE_3M" = "1" ] && command -v git >/dev/null 2>&1; then
    echo "3M-Diffusion data not found; cloning reference repo:"
    echo "  url=$THREE_M_GIT_URL"
    echo "  dest=$THREE_M_ROOT"
    mkdir -p "$(dirname "$THREE_M_ROOT")"
    git clone "$THREE_M_GIT_URL" "$THREE_M_ROOT"
  else
    echo "Missing 3M-Diffusion data directory: $THREE_M_ROOT/data" >&2
    echo "Set SUCC_3M_ROOT or enable SUCC_AUTO_CLONE_3M=1." >&2
    exit 2
  fi
fi

if [ ! -f "$EDIT_MANIFEST" ]; then
  echo "Missing edit manifest: $EDIT_MANIFEST" >&2
  echo "Build the multiproperty dataset first, then rerun this script." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/export_unified_condition_dataset.py" \
  --three-m-root "$THREE_M_ROOT" \
  --edit-manifest "$EDIT_MANIFEST" \
  --description-limit-per-split "$DESCRIPTION_LIMIT" \
  --edit-limit "$EDIT_LIMIT" \
  --output-dir "$OUTPUT_DIR/dataset"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_alignment_pretraining.py" \
  --train-jsonl "$OUTPUT_DIR/dataset/unified_condition_train.jsonl" \
  --output-dir "$OUTPUT_DIR/alignment" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --limit "$TRAIN_LIMIT"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_edit_condition_tokens.py" \
  --train-jsonl "$OUTPUT_DIR/dataset/unified_condition_train.jsonl" \
  --output-dir "$OUTPUT_DIR/edit_condition_tokens" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --limit "$TRAIN_LIMIT" \
  --export-features

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_latent_diffusion_generation.py" \
  --train-jsonl "$OUTPUT_DIR/dataset/unified_condition_train.jsonl" \
  --condition-connector "$OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt" \
  --output-dir "$OUTPUT_DIR/latent_diffusion" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --limit "$TRAIN_LIMIT" \
  --timesteps 20

"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_latent_diffusion_generation.py" \
  --eval-jsonl "$OUTPUT_DIR/dataset/unified_condition_eval.jsonl" \
  --condition-connector "$OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt" \
  --diffusion-checkpoint "$OUTPUT_DIR/latent_diffusion/latent_diffusion_generation.pt" \
  --output-dir "$OUTPUT_DIR/eval_latent" \
  --limit "$EVAL_LIMIT" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --sample-steps "$EVAL_SAMPLE_STEPS"

echo "Unified smoke finished:"
echo "  dataset=$OUTPUT_DIR/dataset/summary.json"
echo "  alignment=$OUTPUT_DIR/alignment/alignment_model.pt"
echo "  connector=$OUTPUT_DIR/edit_condition_tokens/edit_condition_connector.pt"
echo "  diffusion=$OUTPUT_DIR/latent_diffusion/latent_diffusion_generation.pt"
echo "  eval=$OUTPUT_DIR/eval_latent/metrics.json"
