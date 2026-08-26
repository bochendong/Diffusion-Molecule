#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p23_explicit_task_stage1_v2"
ACCOUNT="${P23_ACCOUNT:-def-hup-ab}"
GRES="${P23_GRES:-gpu:h100:1}"
mkdir -p "$LOG_DIR"
prepare=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-edit500p --time=01:30:00 \
  --cpus-per-task=4 --mem=32G --output="$LOG_DIR/edit500-prepare-%j.log" \
  --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_edit500_prepare.sh")
generate=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-edit500g --time=03:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GRES" --dependency="afterok:$prepare" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/edit500-generate-%j.log" --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_edit500_generate.sh")
score=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-edit500s --time=01:00:00 \
  --cpus-per-task=4 --mem=24G --dependency="afterok:$generate" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/edit500-score-%j.log" --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_edit500_score.sh")
printf 'prepare_job=%s\ngenerate_job=%s\nscore_job=%s\n' "$prepare" "$generate" "$score"
