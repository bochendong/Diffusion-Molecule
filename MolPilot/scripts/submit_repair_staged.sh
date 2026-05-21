#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export MOLPILOT_TASK_MODE=repair
export MOLPILOT_LIMIT="${MOLPILOT_LIMIT:-10000}"
export MOLPILOT_EVAL_LIMIT="${MOLPILOT_EVAL_LIMIT:-1000}"
export MOLPILOT_EVAL_TASKS="${MOLPILOT_EVAL_TASKS:-repair}"
export MOLPILOT_REPAIR_CORRUPTIONS_PER_MOLECULE="${MOLPILOT_REPAIR_CORRUPTIONS_PER_MOLECULE:-2}"
export MOLPILOT_RUN_NAME="${MOLPILOT_RUN_NAME:-molpilot_repair_${MOLPILOT_LIMIT}_$(date +%Y%m%d_%H%M%S)}"
export MOLPILOT_SLURM_TIME="${MOLPILOT_SLURM_TIME:-04:00:00}"

echo "Submitting MolPilot-R repair run:"
echo "  run_name=$MOLPILOT_RUN_NAME"
echo "  limit=$MOLPILOT_LIMIT"
echo "  eval_limit=$MOLPILOT_EVAL_LIMIT"
echo "  repair_corruptions_per_molecule=$MOLPILOT_REPAIR_CORRUPTIONS_PER_MOLECULE"

bash scripts/submit_chembl_staged.sh
