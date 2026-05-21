#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEFAULT_SERVER_PYTHON="/scratch/bdong/venvs/phystabmol/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_SERVER_PYTHON" ]]; then
    PYTHON_BIN="$DEFAULT_SERVER_PYTHON"
  else
    PYTHON_BIN="$(command -v python || command -v python3)"
  fi
fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if ! "$PYTHON_BIN" -c "import numpy" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN cannot import numpy."
  echo "Set PYTHON_BIN=/scratch/bdong/venvs/phystabmol/bin/python or activate the right venv."
  exit 2
fi

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
if [[ "${MOLPILOT_DISABLE_SOURCE_GUIDANCE:-0}" == "1" ]]; then
  SAMPLE_ARGS+=(--disable-source-guidance)
fi
if [[ "${MOLPILOT_DISABLE_GRAPH_EDITOR:-0}" == "1" ]]; then
  SAMPLE_ARGS+=(--disable-graph-editor)
fi

echo "MolPilot resample existing stage"
echo "  stage_root=$STAGE_ROOT"
echo "  data=$DATA"
echo "  sample_dir=$SAMPLE_DIR"
echo "  python_bin=$PYTHON_BIN"
echo "  disable_verifier_ranking=${MOLPILOT_DISABLE_VERIFIER_RANKING:-0}"

"$PYTHON_BIN" -m molpilot.sample \
  --data "$DATA" \
  --autoencoder-dir "$STAGE_ROOT/stage1_autoencoder" \
  --alignment-dir "$STAGE_ROOT/stage2_understanding" \
  --diffusion-dir "$STAGE_ROOT/stage3_diffusion" \
  --output-dir "$SAMPLE_DIR" \
  --limit "${MOLPILOT_EVAL_MOLECULE_LIMIT:-${MOLPILOT_LIMIT:-10000}}" \
  --condition-dim "${MOLPILOT_CONDITION_DIM:-256}" \
  --samples-per-request "${MOLPILOT_SAMPLES:-8}" \
  --decode-top-k "${MOLPILOT_DECODE_TOP_K:-4}" \
  --source-edit-strengths "${MOLPILOT_SOURCE_EDIT_STRENGTHS:-0.25,0.50}" \
  --source-neighborhood-k "${MOLPILOT_SOURCE_NEIGHBORHOOD_K:-32}" \
  --graph-edit-limit "${MOLPILOT_GRAPH_EDIT_LIMIT:-96}" \
  --scaffold-library-k "${MOLPILOT_SCAFFOLD_LIBRARY_K:-32}" \
  --max-requests-per-task "${MOLPILOT_MAX_REQUESTS_PER_TASK:-${MOLPILOT_EVAL_LIMIT:-1000}}" \
  --tasks "${MOLPILOT_EVAL_TASKS:-edit,inpaint,de_novo}" \
  "${SAMPLE_ARGS[@]}" \
  "${RENDER_ARGS[@]}"

"$PYTHON_BIN" -m molpilot.evaluate \
  --candidates "$SAMPLE_DIR/tables/candidates.csv" \
  --out "$STAGE_ROOT/eval_metrics_$OUT_SUFFIX.json"

echo "MolPilot resample finished: $SAMPLE_DIR"
