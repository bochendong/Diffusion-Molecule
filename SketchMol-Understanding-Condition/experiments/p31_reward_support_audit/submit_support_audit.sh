#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P31_OUTPUT_ROOT:-$PROJECT/outputs/p31_reward_support_audit/seed_31001}"
LOG_DIR="$PROJECT/logs/p31_reward_support_audit"
GPU="${P31_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"
preflight=$(sbatch --parsable --account=def-hup-ab --job-name=p31-check --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --output="$LOG_DIR/preflight-%j.log" \
  --export=ALL,P31_SCRIPT_DIR="$SCRIPT_DIR",P31_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_preflight.sh")
prepare=$(sbatch --parsable --account=def-hup-ab --job-name=p31-data --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$preflight" --output="$LOG_DIR/prepare-%j.log" \
  --export=ALL,P31_SCRIPT_DIR="$SCRIPT_DIR",P31_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_prepare.sh")
jobs=()
for shard in 0 1 2 3; do
  job=$(sbatch --parsable --account=def-hup-ab --job-name="p31-s$shard" --time=02:00:00 \
    --cpus-per-task=4 --mem=48G --gres="$GPU" --dependency="afterok:$prepare" \
    --output="$LOG_DIR/shard${shard}-%j.log" \
    --export=ALL,P31_SCRIPT_DIR="$SCRIPT_DIR",P31_OUTPUT_ROOT="$OUT",P31_SHARD="$shard" "$SCRIPT_DIR/run_shard.sh")
  jobs+=("$job")
done
dependency=$(IFS=:; echo "${jobs[*]}")
merge=$(sbatch --parsable --account=def-hup-ab --job-name=p31-merge --time=00:10:00 \
  --cpus-per-task=2 --mem=8G --dependency="afterok:$dependency" --output="$LOG_DIR/merge-%j.log" \
  --export=ALL,P31_SCRIPT_DIR="$SCRIPT_DIR",P31_OUTPUT_ROOT="$OUT" "$SCRIPT_DIR/run_merge.sh")
printf 'preflight_job=%s\nprepare_job=%s\nshard_jobs=%s\nmerge_job=%s\n' \
  "$preflight" "$prepare" "${jobs[*]}" "$merge"
