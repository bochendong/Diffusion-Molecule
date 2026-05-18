#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

"$PYTHON_BIN" -m molpilot.experiment \
  --run-name "${MOLPILOT_RUN_NAME:-molpilot_server_v1}" \
  --data "${MOLPILOT_DATA:-data/molecules.csv}" \
  --limit "${MOLPILOT_LIMIT:-100000}" \
  --epochs "${MOLPILOT_EPOCHS:-20}" \
  --timesteps "${MOLPILOT_TIMESTEPS:-100}" \
  --batch-size "${MOLPILOT_BATCH_SIZE:-512}" \
  --hidden-dim "${MOLPILOT_HIDDEN_DIM:-512}" \
  --samples-per-request "${MOLPILOT_SAMPLES:-8}" \
  --decode-top-k "${MOLPILOT_DECODE_TOP_K:-4}"

