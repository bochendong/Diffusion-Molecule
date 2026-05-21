#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RUN_NAME="${MOLPILOT_RUN_NAME:-staged_smoke}"
STAGE_ROOT="${MOLPILOT_STAGE_ROOT:-outputs/stages/$RUN_NAME}"
DATA="${MOLPILOT_DATA:-}"
LIMIT="${MOLPILOT_LIMIT:-0}"
TASK_MODE="${MOLPILOT_TASK_MODE:-verified}"

AE_DIR="$STAGE_ROOT/stage1_autoencoder"
ALIGN_DIR="$STAGE_ROOT/stage2_understanding"
DIFF_DIR="$STAGE_ROOT/stage3_diffusion"
SAMPLE_DIR="$STAGE_ROOT/stage4_samples"

echo "MolPilot staged smoke run"
echo "  stage_root=$STAGE_ROOT"
echo "  task_mode=$TASK_MODE"

"$PYTHON_BIN" -m molpilot.train_autoencoder \
  --data "$DATA" \
  --output-dir "$AE_DIR" \
  --limit "$LIMIT" \
  --codec "${MOLPILOT_CODEC:-sequence}" \
  --representation "${MOLPILOT_REPRESENTATION:-auto}" \
  --embedding-dim "${MOLPILOT_EMBEDDING_DIM:-128}" \
  --max-length "${MOLPILOT_MAX_LENGTH:-96}" \
  --epochs "${MOLPILOT_AE_EPOCHS:-2}" \
  --latent-dim "${MOLPILOT_LATENT_DIM:-64}"

"$PYTHON_BIN" -m molpilot.train_understanding \
  --data "$DATA" \
  --autoencoder-dir "$AE_DIR" \
  --output-dir "$ALIGN_DIR" \
  --limit "$LIMIT" \
  --task-mode "$TASK_MODE" \
  --repair-corruptions "${MOLPILOT_REPAIR_CORRUPTIONS:-}" \
  --repair-corruptions-per-molecule "${MOLPILOT_REPAIR_CORRUPTIONS_PER_MOLECULE:-2}" \
  --epochs "${MOLPILOT_ALIGN_EPOCHS:-2}" \
  --model-kind "${MOLPILOT_STAGE2_MODEL:-jepa}"

"$PYTHON_BIN" -m molpilot.train_diffusion \
  --data "$DATA" \
  --autoencoder-dir "$AE_DIR" \
  --alignment-dir "$ALIGN_DIR" \
  --output-dir "$DIFF_DIR" \
  --limit "$LIMIT" \
  --task-mode "$TASK_MODE" \
  --repair-corruptions "${MOLPILOT_REPAIR_CORRUPTIONS:-}" \
  --repair-corruptions-per-molecule "${MOLPILOT_REPAIR_CORRUPTIONS_PER_MOLECULE:-2}" \
  --epochs "${MOLPILOT_DIFFUSION_EPOCHS:-2}" \
  --timesteps "${MOLPILOT_TIMESTEPS:-20}"

"$PYTHON_BIN" -m molpilot.sample \
  --data "$DATA" \
  --autoencoder-dir "$AE_DIR" \
  --alignment-dir "$ALIGN_DIR" \
  --diffusion-dir "$DIFF_DIR" \
  --output-dir "$SAMPLE_DIR" \
  --limit "$LIMIT" \
  --task-mode "$TASK_MODE" \
  --repair-corruptions "${MOLPILOT_REPAIR_CORRUPTIONS:-}" \
  --repair-corruptions-per-molecule "${MOLPILOT_REPAIR_CORRUPTIONS_PER_MOLECULE:-2}" \
  --samples-per-request "${MOLPILOT_SAMPLES:-2}" \
  --decode-top-k "${MOLPILOT_DECODE_TOP_K:-2}" \
  --tasks "${MOLPILOT_EVAL_TASKS:-edit,inpaint,de_novo,repair}"

"$PYTHON_BIN" -m molpilot.evaluate \
  --candidates "$SAMPLE_DIR/tables/candidates.csv" \
  --out "$STAGE_ROOT/eval_metrics.json"

echo "MolPilot staged smoke finished: $STAGE_ROOT"
