#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$PROJECT/logs/p25_p23_joint_group_rl"
mkdir -p "$LOG_DIR"
cp13=$(sbatch --parsable --account=def-hup-ab --job-name=p25-gate-c13 --time=00:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --output="$LOG_DIR/checkpoint-013-%j.log" \
  --export=ALL,P25_SCRIPT_DIR="$SCRIPT_DIR",P25_CHECKPOINT_INDEX=013 "$SCRIPT_DIR/run_checkpoint_eval.sh")
cp26=$(sbatch --parsable --account=def-hup-ab --job-name=p25-gate-c26 --time=00:45:00 --cpus-per-task=4 --mem=48G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --output="$LOG_DIR/checkpoint-026-%j.log" \
  --export=ALL,P25_SCRIPT_DIR="$SCRIPT_DIR",P25_CHECKPOINT_INDEX=026 "$SCRIPT_DIR/run_checkpoint_eval.sh")
compare=$(sbatch --parsable --account=def-hup-ab --job-name=p25-ckpt-compare --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$cp13:$cp26" --output="$LOG_DIR/checkpoint-compare-%j.log" \
  --export=ALL,P25_SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/run_checkpoint_compare.sh")
printf 'checkpoint_013_job=%s\ncheckpoint_026_job=%s\ncompare_job=%s\n' "$cp13" "$cp26" "$compare"
