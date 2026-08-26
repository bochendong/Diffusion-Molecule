#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p24_molprogram_instruct_4m"
ACCOUNT="${P24_ACCOUNT:-def-hup-ab}"
mkdir -p "$LOG_DIR"

download=$(sbatch --parsable --account="$ACCOUNT" --job-name=p24-download --time=04:00:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/download-%j.log" \
  --export=ALL,P24_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/download_pubchem.sh")
extract=$(sbatch --parsable --account="$ACCOUNT" --job-name=p24-extract --time=03:00:00 \
  --array=0-7 --cpus-per-task=4 --mem=24G --dependency="afterok:$download" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/extract-%A_%a.log" --export=ALL,P24_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/extract_pubchem_array.sh")
release=$(sbatch --parsable --account="$ACCOUNT" --job-name=p24-release --time=06:00:00 \
  --cpus-per-task=8 --mem=96G --dependency="afterok:$extract" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/release-%j.log" --export=ALL,P24_SCRIPT_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/build_release_job.sh")
printf 'download_job=%s\nextract_array_job=%s\nrelease_job=%s\n' "$download" "$extract" "$release"

