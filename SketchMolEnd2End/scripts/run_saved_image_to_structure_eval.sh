#!/usr/bin/env bash
# Re-evaluate a saved no-OCR image-to-structure model with expanded candidates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${SKETCHMOL_E2E_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
RUN_DIR="${SKETCHMOL_E2E_RUN_DIR:-SketchMolEnd2End/outputs/runs/image_to_structure_fp_aux_seed7}"
PAIR_DIR="${SKETCHMOL_E2E_PAIR_DIR:-SketchSMILES/outputs/pairs/phys_50k}"
BEAM_SIZE="${SKETCHMOL_E2E_BEAM_SIZE:-32}"
DECODING="${SKETCHMOL_E2E_DECODING:-beam}"
RERANK_MODE="${SKETCHMOL_E2E_RERANK_MODE:-beam}"
OUTPUT_DIR="${SKETCHMOL_E2E_OUTPUT_DIR:-${RUN_DIR}_eval_${DECODING}_beam${BEAM_SIZE}}"
LENGTH_PENALTY="${SKETCHMOL_E2E_LENGTH_PENALTY:-0.0}"
SAMPLES_PER_CONDITION="${SKETCHMOL_E2E_SAMPLES_PER_CONDITION:-32}"
TEMPERATURE="${SKETCHMOL_E2E_TEMPERATURE:-0.9}"
SAMPLE_TOP_K="${SKETCHMOL_E2E_SAMPLE_TOP_K:-16}"
IMAGE_SIZE="${SKETCHMOL_E2E_IMAGE_SIZE:-128}"
SAMPLE_COUNT="${SKETCHMOL_E2E_SAMPLE_COUNT:-64}"
EVAL_LIMIT="${SKETCHMOL_E2E_EVAL_LIMIT:-}"
SEED="${SKETCHMOL_E2E_SEED:-7}"
DEVICE="${SKETCHMOL_E2E_DEVICE:-auto}"

export PYTHONPATH="$PROJECT_DIR:$REPO_ROOT/SketchSMILES${PYTHONPATH:+:$PYTHONPATH}"
export SKETCHMOL_E2E_MODULES="${SKETCHMOL_E2E_MODULES:-gcc rdkit/2025.09.4}"

if [[ -n "${SKETCHMOL_E2E_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_E2E_MODULES
fi

echo "SketchMolEnd2End saved-model eval"
echo "  python=$PYTHON_BIN"
echo "  run_dir=$RUN_DIR"
echo "  pair_dir=$PAIR_DIR"
echo "  output_dir=$OUTPUT_DIR"
echo "  decoding=$DECODING"
echo "  beam_size=$BEAM_SIZE"
echo "  rerank_mode=$RERANK_MODE"
echo "  image_size=$IMAGE_SIZE"
echo "  eval_limit=${EVAL_LIMIT:-<all>}"
echo "  device=$DEVICE"

if [[ ! -f "$RUN_DIR/model.pt" ]]; then
  echo "ERROR: model.pt not found under $RUN_DIR" >&2
  exit 2
fi
if [[ ! -f "$RUN_DIR/vocab.json" ]]; then
  echo "ERROR: vocab.json not found under $RUN_DIR" >&2
  exit 2
fi

ARGS=(
  -m sketchmol_end2end.evaluate_saved_image_to_structure
  --run-dir "$RUN_DIR"
  --output-dir "$OUTPUT_DIR"
  --pair-dir "$PAIR_DIR"
  --decoding "$DECODING"
  --beam-size "$BEAM_SIZE"
  --length-penalty "$LENGTH_PENALTY"
  --rerank-mode "$RERANK_MODE"
  --samples-per-condition "$SAMPLES_PER_CONDITION"
  --temperature "$TEMPERATURE"
  --sample-top-k "$SAMPLE_TOP_K"
  --image-size "$IMAGE_SIZE"
  --sample-count "$SAMPLE_COUNT"
  --seed "$SEED"
  --device "$DEVICE"
)
if [[ -n "$EVAL_LIMIT" ]]; then
  ARGS+=(--eval-limit "$EVAL_LIMIT")
fi

"$PYTHON_BIN" "${ARGS[@]}"

echo
echo "Saved-model eval finished: $OUTPUT_DIR"
echo "  metrics=$OUTPUT_DIR/metrics.json"
echo "  predictions=$OUTPUT_DIR/predictions.csv"
