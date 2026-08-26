#!/usr/bin/env bash
set -euo pipefail

dependency="${1:?usage: submit_table1_eval.sh refresh_job_id}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p24_molprogram_instruct_4m"
ACCOUNT="${P24_ACCOUNT:-def-hup-ab}"
GRES="${P24_GRES:-gpu:h100:1}"
mkdir -p "$LOG_DIR"
generation=$(sbatch --parsable --account="$ACCOUNT" --job-name=p24-table1g --time=06:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GRES" --dependency="afterok:$dependency" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/table1-generate-%j.log" --export=ALL,P24_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/run_table1_generate.sh")
scoring=$(sbatch --parsable --account="$ACCOUNT" --job-name=p24-table1s --time=01:00:00 \
  --cpus-per-task=4 --mem=24G --dependency="afterok:$generation" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/table1-score-%j.log" --export=ALL,P24_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/run_table1_finalize.sh")
printf 'table1_generation_job=%s\ntable1_scoring_job=%s\n' "$generation" "$scoring"
