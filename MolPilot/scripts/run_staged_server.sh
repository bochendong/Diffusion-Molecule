#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RUN_NAME="${MOLPILOT_RUN_NAME:-molpilot_staged_v1}"
STAGE_ROOT="${MOLPILOT_STAGE_ROOT:-outputs/stages/$RUN_NAME}"
DATA="${MOLPILOT_DATA:-../PhysTabMol/data/molecules.csv}"
LIMIT="${MOLPILOT_LIMIT:-100000}"

AE_DIR="$STAGE_ROOT/stage1_autoencoder"
ALIGN_DIR="$STAGE_ROOT/stage2_understanding"
DIFF_DIR="$STAGE_ROOT/stage3_diffusion"
SAMPLE_DIR="$STAGE_ROOT/stage4_samples"
RENDER_ARGS=()
if [[ "${MOLPILOT_RENDER_MISSING_IMAGES:-0}" == "1" ]]; then
  RENDER_ARGS+=(--render-missing-images)
fi
SAMPLE_ARGS=()
if [[ "${MOLPILOT_DISABLE_VERIFIER_RANKING:-0}" == "1" ]]; then
  SAMPLE_ARGS+=(--disable-verifier-ranking)
fi

echo "MolPilot staged server run"
echo "  data=$DATA"
echo "  limit=$LIMIT"
echo "  stage_root=$STAGE_ROOT"

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: MOLPILOT_DATA points to '$DATA', but that file does not exist."
  echo "For the ChEMBL file created by PhysTabMol, run from MolPilot with:"
  echo "  MOLPILOT_DATA=../PhysTabMol/data/molecules.csv bash scripts/run_staged_server.sh"
  echo "Use scripts/run_staged_smoke.sh for the built-in 6-molecule smoke test."
  exit 2
fi

DATA_ROWS=$(wc -l < "$DATA" | tr -d ' ')
if [[ "$DATA_ROWS" -lt 1000 ]]; then
  echo "WARNING: '$DATA' has only $DATA_ROWS lines. This is probably a small/debug run."
fi

"$PYTHON_BIN" -m molpilot.train_autoencoder \
  --data "$DATA" \
  --output-dir "$AE_DIR" \
  --limit "$LIMIT" \
  --codec "${MOLPILOT_CODEC:-sequence}" \
  --representation "${MOLPILOT_REPRESENTATION:-auto}" \
  --feature-dim "${MOLPILOT_FEATURE_DIM:-256}" \
  --latent-dim "${MOLPILOT_LATENT_DIM:-64}" \
  --embedding-dim "${MOLPILOT_EMBEDDING_DIM:-192}" \
  --max-length "${MOLPILOT_MAX_LENGTH:-128}" \
  --hidden-dim "${MOLPILOT_AE_HIDDEN_DIM:-768}" \
  --layers "${MOLPILOT_AE_LAYERS:-4}" \
  --epochs "${MOLPILOT_AE_EPOCHS:-60}" \
  --batch-size "${MOLPILOT_AE_BATCH_SIZE:-512}"

"$PYTHON_BIN" -m molpilot.train_understanding \
  --data "$DATA" \
  --autoencoder-dir "$AE_DIR" \
  --output-dir "$ALIGN_DIR" \
  --limit "$LIMIT" \
  --condition-dim "${MOLPILOT_CONDITION_DIM:-256}" \
  --hidden-dim "${MOLPILOT_ALIGN_HIDDEN_DIM:-768}" \
  --layers "${MOLPILOT_ALIGN_LAYERS:-4}" \
  --epochs "${MOLPILOT_ALIGN_EPOCHS:-60}" \
  --batch-size "${MOLPILOT_ALIGN_BATCH_SIZE:-1024}" \
  --contrastive-weight "${MOLPILOT_CONTRASTIVE_WEIGHT:-0.05}" \
  --delta-weight "${MOLPILOT_JEPA_DELTA_WEIGHT:-0.25}" \
  --sigreg-weight "${MOLPILOT_JEPA_SIGREG_WEIGHT:-0.01}" \
  --model-kind "${MOLPILOT_STAGE2_MODEL:-jepa}" \
  "${RENDER_ARGS[@]}"

"$PYTHON_BIN" -m molpilot.train_diffusion \
  --data "$DATA" \
  --autoencoder-dir "$AE_DIR" \
  --alignment-dir "$ALIGN_DIR" \
  --output-dir "$DIFF_DIR" \
  --limit "$LIMIT" \
  --condition-dim "${MOLPILOT_CONDITION_DIM:-256}" \
  --hidden-dim "${MOLPILOT_DIFFUSION_HIDDEN_DIM:-1024}" \
  --layers "${MOLPILOT_DIFFUSION_LAYERS:-6}" \
  --epochs "${MOLPILOT_DIFFUSION_EPOCHS:-100}" \
  --batch-size "${MOLPILOT_DIFFUSION_BATCH_SIZE:-1024}" \
  --timesteps "${MOLPILOT_TIMESTEPS:-100}" \
  "${RENDER_ARGS[@]}"

"$PYTHON_BIN" -m molpilot.sample \
  --data "$DATA" \
  --autoencoder-dir "$AE_DIR" \
  --alignment-dir "$ALIGN_DIR" \
  --diffusion-dir "$DIFF_DIR" \
  --output-dir "$SAMPLE_DIR" \
  --limit "${MOLPILOT_EVAL_LIMIT:-5000}" \
  --condition-dim "${MOLPILOT_CONDITION_DIM:-256}" \
  --samples-per-request "${MOLPILOT_SAMPLES:-8}" \
  --decode-top-k "${MOLPILOT_DECODE_TOP_K:-4}" \
  "${SAMPLE_ARGS[@]}" \
  "${RENDER_ARGS[@]}"

"$PYTHON_BIN" -m molpilot.evaluate \
  --candidates "$SAMPLE_DIR/tables/candidates.csv" \
  --out "$STAGE_ROOT/eval_metrics.json"

echo "MolPilot staged server run finished: $STAGE_ROOT"
