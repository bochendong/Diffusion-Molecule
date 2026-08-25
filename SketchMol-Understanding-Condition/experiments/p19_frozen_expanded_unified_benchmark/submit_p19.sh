#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p19_frozen_expanded_unified_benchmark"
mkdir -p "$LOG_DIR"
prepare=$(sbatch --parsable --account=def-hup-ab \
  --job-name=p19-freeze --time=00:15:00 --cpus-per-task=4 --mem=16G \
  --output="$LOG_DIR/prepare-%j.log" --export=ALL,P19_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_prepare.sh")
p17=$(sbatch --parsable --account=def-hup-ab \
  --job-name=p19-p17-gen --time=01:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$prepare" \
  --output="$LOG_DIR/p17-%j.log" --export=ALL,P19_SCRIPT_DIR="$SCRIPT_DIR",P19_MODEL=p17 "$SCRIPT_DIR/run_generate.sh")
p18=$(sbatch --parsable --account=def-hup-ab \
  --job-name=p19-p18-gen --time=01:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --dependency="afterok:$prepare" \
  --output="$LOG_DIR/p18-%j.log" --export=ALL,P19_SCRIPT_DIR="$SCRIPT_DIR",P19_MODEL=p18 "$SCRIPT_DIR/run_generate.sh")
final=$(sbatch --parsable --account=def-hup-ab \
  --job-name=p19-final --time=00:45:00 --cpus-per-task=4 --mem=24G \
  --dependency="afterok:$p17:$p18" --output="$LOG_DIR/final-%j.log" \
  --export=ALL,P19_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_finalize.sh")
printf 'prepare_job=%s\np17_generation_job=%s\np18_generation_job=%s\nfinal_job=%s\n' "$prepare" "$p17" "$p18" "$final"
