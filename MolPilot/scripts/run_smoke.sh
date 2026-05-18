#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

"$PYTHON_BIN" -m molpilot.experiment \
  --run-name "${MOLPILOT_RUN_NAME:-smoke}" \
  --data "${MOLPILOT_DATA:-}" \
  --limit "${MOLPILOT_LIMIT:-0}" \
  --epochs "${MOLPILOT_EPOCHS:-2}" \
  --timesteps "${MOLPILOT_TIMESTEPS:-20}" \
  --samples-per-request "${MOLPILOT_SAMPLES:-2}" \
  --decode-top-k "${MOLPILOT_DECODE_TOP_K:-2}" \
  --disable-render-missing-images

