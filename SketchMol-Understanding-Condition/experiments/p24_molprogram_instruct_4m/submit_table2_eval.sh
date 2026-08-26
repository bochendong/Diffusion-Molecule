#!/usr/bin/env bash
set -euo pipefail

dependency="${1:?usage: submit_table2_eval.sh refresh_job_id}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p24_molprogram_instruct_4m"
ACCOUNT="${P24_ACCOUNT:-def-hup-ab}"
GRES="${P24_GRES:-gpu:h100:1}"
mkdir -p "$LOG_DIR"
generation=$(sbatch --parsable --account="$ACCOUNT" --job-name=p24-table2g --time=03:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GRES" --dependency="afterok:$dependency" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/table2-generate-%j.log" --export=ALL,P24_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/run_table2_generate.sh")
scoring=$(sbatch --parsable --account="$ACCOUNT" --job-name=p24-table2s --time=01:00:00 \
  --cpus-per-task=4 --mem=24G --dependency="afterok:$generation" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/table2-score-%j.log" --export=ALL,P24_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/run_table2_score.sh")
printf 'table2_generation_job=%s\ntable2_scoring_job=%s\n' "$generation" "$scoring"
