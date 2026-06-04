#!/usr/bin/env bash
# Export condition features/query tokens for mixed-objective baseline rows.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023
module load python/3.11
module load rdkit/2025.09.4

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python}}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

BASELINE_CSV="${SUCC_BASELINE_CSV:-SketchMol-Understanding-Condition/outputs/mixed_objective_dataset_8k_strict_v2/baseline_variants.csv}"
ENCODER="${SUCC_ENCODER:-proxy}"
VARIANTS="${SUCC_VARIANTS:-}"
OUTPUT_DIR="${SUCC_OUTPUT_DIR:-SketchMol-Understanding-Condition/outputs/condition_features_mixed_v2_${ENCODER}}"
LIMIT="${SUCC_LIMIT:-}"
IMAGE_ENCODER_CHECKPOINT="${SUCC_IMAGE_ENCODER_CHECKPOINT:-}"
POOLED_DIM="${SUCC_POOLED_DIM:-768}"
NUM_QUERIES="${SUCC_NUM_QUERIES:-16}"
QUERY_DIM="${SUCC_QUERY_DIM:-256}"
HF_MODEL_NAME_OR_PATH="${SUCC_HF_MODEL_NAME_OR_PATH:-}"
HF_DEVICE_MAP="${SUCC_HF_DEVICE_MAP:-auto}"
HF_DTYPE="${SUCC_HF_DTYPE:-auto}"
HF_BATCH_SIZE="${SUCC_HF_BATCH_SIZE:-1}"
HF_MAX_LENGTH="${SUCC_HF_MAX_LENGTH:-2048}"
HF_ATTN_IMPLEMENTATION="${SUCC_HF_ATTN_IMPLEMENTATION:-}"
HF_PROMPT_STYLE="${SUCC_HF_PROMPT_STYLE:-auto}"
HF_RENDER_IMAGE_SIZE="${SUCC_HF_RENDER_IMAGE_SIZE:-256}"
HF_TRUST_REMOTE_CODE="${SUCC_HF_TRUST_REMOTE_CODE:-1}"

echo "Exporting condition encoder features"
echo "  python=$PYTHON_BIN"
echo "  baseline_csv=$BASELINE_CSV"
echo "  encoder=$ENCODER"
if [[ -n "$VARIANTS" ]]; then
  echo "  variants=$VARIANTS"
fi
echo "  output_dir=$OUTPUT_DIR"
echo "  pooled_dim=$POOLED_DIM"
echo "  num_queries=$NUM_QUERIES"
echo "  query_dim=$QUERY_DIM"
if [[ -n "$LIMIT" ]]; then
  echo "  limit=$LIMIT"
fi
if [[ -n "$IMAGE_ENCODER_CHECKPOINT" ]]; then
  echo "  image_encoder_checkpoint=$IMAGE_ENCODER_CHECKPOINT"
fi
if [[ "$ENCODER" == "hf_vlm" ]]; then
  echo "  hf_model_name_or_path=$HF_MODEL_NAME_OR_PATH"
  echo "  hf_device_map=$HF_DEVICE_MAP"
  echo "  hf_dtype=$HF_DTYPE"
  echo "  hf_batch_size=$HF_BATCH_SIZE"
  echo "  hf_prompt_style=$HF_PROMPT_STYLE"
  echo "  hf_render_image_size=$HF_RENDER_IMAGE_SIZE"
fi

LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
VARIANT_ARGS=()
if [[ -n "$VARIANTS" ]]; then
  VARIANT_ARGS=(--variants "$VARIANTS")
fi
CHECKPOINT_ARGS=()
if [[ -n "$IMAGE_ENCODER_CHECKPOINT" ]]; then
  CHECKPOINT_ARGS=(--image-encoder-checkpoint "$IMAGE_ENCODER_CHECKPOINT")
fi
HF_ARGS=()
if [[ "$ENCODER" == "hf_vlm" ]]; then
  if [[ -z "$HF_MODEL_NAME_OR_PATH" ]]; then
    echo "ERROR: SUCC_HF_MODEL_NAME_OR_PATH is required when SUCC_ENCODER=hf_vlm" >&2
    exit 2
  fi
  HF_ARGS=(
    --hf-model-name-or-path "$HF_MODEL_NAME_OR_PATH"
    --hf-device-map "$HF_DEVICE_MAP"
    --hf-dtype "$HF_DTYPE"
    --hf-batch-size "$HF_BATCH_SIZE"
    --hf-max-length "$HF_MAX_LENGTH"
    --hf-prompt-style "$HF_PROMPT_STYLE"
    --hf-render-image-size "$HF_RENDER_IMAGE_SIZE"
  )
  if [[ -n "$HF_ATTN_IMPLEMENTATION" ]]; then
    HF_ARGS+=(--hf-attn-implementation "$HF_ATTN_IMPLEMENTATION")
  fi
  if [[ "$HF_TRUST_REMOTE_CODE" == "1" ]]; then
    HF_ARGS+=(--hf-trust-remote-code)
  else
    HF_ARGS+=(--hf-no-trust-remote-code)
  fi
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/export_condition_features.py" \
  --encoder "$ENCODER" \
  "${VARIANT_ARGS[@]}" \
  --baseline-variants-csv "$BASELINE_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --pooled-dim "$POOLED_DIM" \
  --num-queries "$NUM_QUERIES" \
  --query-dim "$QUERY_DIM" \
  "${CHECKPOINT_ARGS[@]}" \
  "${HF_ARGS[@]}" \
  "${LIMIT_ARGS[@]}"

echo "Condition encoder export finished."
