#!/usr/bin/env bash
# Run UniVideo latent inference on a Table1-balanced eval JSONL from a trained checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
UNIFIED_OUTPUT_DIR="${SUCC_UNIFIED_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_moledit_instruct_v2_fix}"
TABLE1_PACK_DIR="${SUCC_TABLE1_PACK_DIR:-$UNIFIED_OUTPUT_DIR/dataset/table1_benchmark}"
EVAL_JSONL="${SUCC_TABLE1_EVAL_JSONL:-$TABLE1_PACK_DIR/table1_eval.jsonl}"
EVAL_OUTPUT_DIR="${SUCC_TABLE1_EVAL_OUTPUT_DIR:-$UNIFIED_OUTPUT_DIR/univideo_molecule/table1_eval_latent}"
RESUME_CHECKPOINT="${SUCC_RESUME_CHECKPOINT:-$UNIFIED_OUTPUT_DIR/univideo_molecule/univideo_molecule_generation.pt}"
FEATURES_DIR="${SUCC_CONDITION_FEATURES_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_moledit_hf_vlm_v2_fix}"
IMAGE_VAE_CHECKPOINT="${SUCC_IMAGE_VAE_CHECKPOINT:-SketchMol-Understanding-Condition/outputs/univideo_molecule_generation_v2_residual_ink/molecule_image_vae/molecule_image_vae.pt}"

LATENT_BACKEND="${SUCC_LATENT_BACKEND:-image_vae}"
EVAL_BATCH_SIZE="${SUCC_EVAL_BATCH_SIZE:-64}"
EVAL_LIMIT="${SUCC_TABLE1_EVAL_LIMIT:-0}"
SAMPLE_STEPS="${SUCC_SAMPLE_STEPS:-20}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$EVAL_JSONL" ]]; then
  echo "ERROR: missing Table1 eval JSONL: $EVAL_JSONL" >&2
  echo "Run export_moledit_table1_benchmark_pack.py first." >&2
  exit 2
fi
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "ERROR: missing checkpoint: $RESUME_CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$EVAL_OUTPUT_DIR"

LATENT_BACKEND_ARGS=()
if [[ "$LATENT_BACKEND" == "image_vae" ]]; then
  LATENT_BACKEND_ARGS+=(--latent-backend image_vae --image-vae-checkpoint "$IMAGE_VAE_CHECKPOINT")
else
  echo "ERROR: table1 eval-only currently supports image_vae only (got $LATENT_BACKEND)." >&2
  exit 2
fi

EVAL_LIMIT_ARGS=()
if [[ "$EVAL_LIMIT" != "0" ]]; then
  EVAL_LIMIT_ARGS+=(--eval-limit "$EVAL_LIMIT")
else
  EVAL_LIMIT_ARGS+=(--eval-limit 100000)
fi

echo "UniVideo Table1 eval-only latent inference"
echo "  python=$PYTHON_BIN"
echo "  eval_jsonl=$EVAL_JSONL"
echo "  output_dir=$EVAL_OUTPUT_DIR"
echo "  resume_checkpoint=$RESUME_CHECKPOINT"
echo "  condition_features_dir=$FEATURES_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/train_univideo_molecule_generation.py" \
  --train-jsonl "$EVAL_JSONL" \
  --eval-jsonl "$EVAL_JSONL" \
  --condition-features-dir "$FEATURES_DIR" \
  --condition-feature-array query_tokens \
  --condition-feature-variant full \
  --output-dir "$EVAL_OUTPUT_DIR" \
  --limit 1 \
  "${EVAL_LIMIT_ARGS[@]}" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --sample-steps "$SAMPLE_STEPS" \
  --eval-only \
  --resume-checkpoint "$RESUME_CHECKPOINT" \
  "${LATENT_BACKEND_ARGS[@]}"

echo
echo "Table1 eval latent ready:"
echo "  metrics=$EVAL_OUTPUT_DIR/eval_latent/metrics.json"
echo "  predictions=$EVAL_OUTPUT_DIR/eval_latent/predictions.csv"
echo "  generated_latents=$EVAL_OUTPUT_DIR/eval_latent/generated_latents.npy"
