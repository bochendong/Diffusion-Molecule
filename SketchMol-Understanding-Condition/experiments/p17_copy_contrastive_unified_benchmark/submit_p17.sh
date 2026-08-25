#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/logs/p17_copy_contrastive_unified_benchmark"
mkdir -p "$LOG_DIR"
export P17_SCRIPT_DIR="$SCRIPT_DIR"
base=$(sbatch --parsable --account=def-hup-ab --job-name=p17-p16-base --time=00:45:00 --cpus-per-task=4 --mem=48G --gres=gpu:h100:1 --output="$LOG_DIR/baseline-%j.log" --export=ALL,P17_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_p17_baseline.sh")
train=$(sbatch --parsable --account=def-hup-ab --job-name=p17-contrast --time=00:45:00 --cpus-per-task=4 --mem=48G --gres=gpu:h100:1 --output="$LOG_DIR/train-%j.log" --export=ALL,P17_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_p17_train.sh")
final=$(sbatch --parsable --account=def-hup-ab --job-name=p17-pilot --time=01:30:00 --cpus-per-task=4 --mem=48G --gres=gpu:h100:1 --dependency="afterok:$base:$train" --output="$LOG_DIR/final-%j.log" --export=ALL,P17_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_p17_finalize.sh")
printf 'baseline_job=%s\ntrain_job=%s\nfinal_job=%s\n' "$base" "$train" "$final"
