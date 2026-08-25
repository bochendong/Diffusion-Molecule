#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p23_explicit_task_stage1_v2"
ACCOUNT="${P23_ACCOUNT:-def-hup-ab}"
GRES="${P23_GRES:-gpu:h100:1}"
mkdir -p "$LOG_DIR"

prepare=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-prepare --time=00:45:00 \
  --cpus-per-task=4 --mem=16G --output="$LOG_DIR/prepare-%j.log" \
  --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_prepare.sh")
sft=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-sft --time=06:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GRES" --dependency="afterok:$prepare" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/sft-%j.log" --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_sft.sh")
contrastive=$(sbatch --parsable --account="$ACCOUNT" --job-name=p23-contrast --time=08:00:00 \
  --cpus-per-task=4 --mem=48G --gres="$GRES" --dependency="afterok:$sft" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/contrast-%j.log" --export=ALL,P23_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_contrastive.sh")
printf 'prepare_job=%s\nsft_job=%s\ncontrastive_job=%s\n' "$prepare" "$sft" "$contrastive"
