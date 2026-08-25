#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p18_validity_aware_multinegative_unified"
mkdir -p "$LOG_DIR"
prepare=$(sbatch --parsable --account=def-hup-ab --job-name=p18-prepare --time=00:10:00 --cpus-per-task=4 --mem=16G --output="$LOG_DIR/prepare-%j.log" --export=ALL,P18_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_p18_prepare.sh")
train=$(sbatch --parsable --account=def-hup-ab --job-name=p18-train --time=00:45:00 --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$prepare" --output="$LOG_DIR/train-%j.log" --export=ALL,P18_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_p18_train.sh")
final=$(sbatch --parsable --account=def-hup-ab --job-name=p18-pilot --time=01:30:00 --cpus-per-task=4 --mem=48G --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$train" --output="$LOG_DIR/final-%j.log" --export=ALL,P18_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_p18_finalize.sh")
printf 'prepare_job=%s\ntrain_job=%s\nfinal_job=%s\n' "$prepare" "$train" "$final"
