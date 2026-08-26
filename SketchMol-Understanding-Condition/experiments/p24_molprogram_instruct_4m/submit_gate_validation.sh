#!/usr/bin/env bash
set -euo pipefail

dependency="${1:?usage: submit_gate_validation.sh gate_job_id}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p24_molprogram_instruct_4m"
ACCOUNT="${P24_ACCOUNT:-def-hup-ab}"
GRES="${P24_GRES:-gpu:h100:1}"
mkdir -p "$LOG_DIR"
job=$(sbatch --parsable --account="$ACCOUNT" --job-name=p24-gateval --time=02:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GRES" --dependency="afterok:$dependency" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/gate-validation-%j.log" --export=ALL,P24_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/run_gate_validation.sh")
printf 'gate_validation_job=%s\n' "$job"
