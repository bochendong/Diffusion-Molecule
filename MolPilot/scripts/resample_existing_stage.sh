#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

STAGE_ROOT="${MOLPILOT_STAGE_ROOT:-${MOLPILOT_RUN_DIR:-}}"
if [[ -z "$STAGE_ROOT" ]]; then
  echo "ERROR: set MOLPILOT_STAGE_ROOT to an existing outputs/stages/<run> directory."
  exit 2
fi

DATA="${MOLPILOT_DATA:-../PhysTabMol/data/molecules.csv}"
OUT_SUFFIX="${MOLPILOT_SAMPLE_SUFFIX:-ranked}"
SAMPLE_DIR="$STAGE_ROOT/stage4_samples_$OUT_SUFFIX"
RENDER_ARGS=()
if [[ "${MOLPILOT_RENDER_MISSING_IMAGES:-0}" == "1" ]]; then
  RENDER_ARGS+=(--render-missing-images)
fi
SAMPLE_ARGS=()
if [[ "${MOLPILOT_DISABLE_VERIFIER_RANKING:-0}" == "1" ]]; then
  SAMPLE_ARGS+=(--disable-verifier-ranking)
fi

echo "MolPilot resample existing stage"
echo "  stage_root=$STAGE_ROOT"
echo "  data=$DATA"
echo "  sample_dir=$SAMPLE_DIR"
echo "  disable_verifier_ranking=${MOLPILOT_DISABLE_VERIFIER_RANKING:-0}"

"$PYTHON_BIN" -m molpilot.sample \
  --data "$DATA" \
  --autoencoder-dir "$STAGE_ROOT/stage1_autoencoder" \
  --alignment-dir "$STAGE_ROOT/stage2_understanding" \
  --diffusion-dir "$STAGE_ROOT/stage3_diffusion" \
  --output-dir "$SAMPLE_DIR" \
  --limit "${MOLPILOT_EVAL_LIMIT:-5000}" \
  --condition-dim "${MOLPILOT_CONDITION_DIM:-256}" \
  --samples-per-request "${MOLPILOT_SAMPLES:-8}" \
  --decode-top-k "${MOLPILOT_DECODE_TOP_K:-4}" \
  "${SAMPLE_ARGS[@]}" \
  "${RENDER_ARGS[@]}"

"$PYTHON_BIN" -m molpilot.evaluate \
  --candidates "$SAMPLE_DIR/tables/candidates.csv" \
  --out "$STAGE_ROOT/eval_metrics_$OUT_SUFFIX.json"

echo "MolPilot resample finished: $SAMPLE_DIR"
