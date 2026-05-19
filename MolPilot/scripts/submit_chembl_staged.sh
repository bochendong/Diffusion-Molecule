#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATA="${MOLPILOT_DATA:-../PhysTabMol/data/molecules.csv}"
LIMIT="${MOLPILOT_LIMIT:-10000}"
EVAL_LIMIT="${MOLPILOT_EVAL_LIMIT:-1000}"
CODEC="${MOLPILOT_CODEC:-sequence}"
RUN_NAME="${MOLPILOT_RUN_NAME:-molpilot_${CODEC}_${LIMIT}_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$DATA" ]]; then
  echo "ERROR: molecule CSV not found: $DATA"
  echo "Expected the ChEMBL file created by PhysTabMol, usually:"
  echo "  ../PhysTabMol/data/molecules.csv"
  exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch was not found. Run this on a Slurm login node."
  exit 2
fi

echo "Submitting MolPilot staged ChEMBL run:"
echo "  data=$DATA"
echo "  limit=$LIMIT"
echo "  eval_limit=$EVAL_LIMIT"
echo "  codec=$CODEC"
echo "  run_name=$RUN_NAME"

export MOLPILOT_DATA="$DATA"
export MOLPILOT_LIMIT="$LIMIT"
export MOLPILOT_EVAL_LIMIT="$EVAL_LIMIT"
export MOLPILOT_CODEC="$CODEC"
export MOLPILOT_RUN_NAME="$RUN_NAME"

sbatch scripts/run_staged_server.slurm.sh
