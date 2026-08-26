#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p23_explicit_task_stage1_v2"
ACCOUNT="${P23_ACCOUNT:-def-hup-ab}"
GRES="${P23_GRES:-gpu:h100:1}"
mkdir -p "$LOG_DIR"

generation=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-paper40 \
  --time=03:00:00 --cpus-per-task=4 --mem=48G --gres="$GRES" \
  --output="$LOG_DIR/paper40-%j.log" --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/run_paper_eval_generate.sh")
scoring=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-paperscore \
  --time=01:00:00 --cpus-per-task=4 --mem=24G --dependency="afterok:$generation" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/paperscore-%j.log" --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/run_paper_eval_finalize.sh")
printf 'generation_job=%s\nscoring_job=%s\n' "$generation" "$scoring"
