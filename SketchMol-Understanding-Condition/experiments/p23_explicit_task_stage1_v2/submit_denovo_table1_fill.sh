#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p23_explicit_task_stage1_v2"
ACCOUNT="${P23_ACCOUNT:-def-hup-ab}"
GRES="${P23_GRES:-gpu:h100:1}"
mkdir -p "$LOG_DIR"
prepare=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-dnfill-p --time=00:30:00 \
  --cpus-per-task=4 --mem=16G --output="$LOG_DIR/denovo-fill-prepare-%j.log" \
  --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_denovo_table1_fill_prepare.sh")
generate=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-dnfill-g --time=03:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GRES" --dependency="afterok:$prepare" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/denovo-fill-generate-%j.log" --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/run_denovo_table1_fill_generate.sh")
score=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-dnfill-s --time=00:30:00 \
  --cpus-per-task=4 --mem=16G --dependency="afterok:$generate" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/denovo-fill-score-%j.log" --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/run_denovo_table1_fill_finalize.sh")
printf 'prepare_job=%s\ngenerate_job=%s\nscore_job=%s\n' "$prepare" "$generate" "$score"
