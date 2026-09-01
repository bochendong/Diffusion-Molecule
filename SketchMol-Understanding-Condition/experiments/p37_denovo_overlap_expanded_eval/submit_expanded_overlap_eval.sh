#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P37_OUTPUT_ROOT:-$PROJECT/outputs/p37_denovo_overlap_expanded_eval/seed_37101}"
LOG_DIR="$PROJECT/logs/p37_denovo_overlap_expanded_eval"
ACCOUNT="${P37_ACCOUNT:-def-hup-ab}"
GPU="${P37_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
mkdir -p "$LOG_DIR"
common="P37_SCRIPT_DIR=$SCRIPT_DIR,P37_OUTPUT_ROOT=$OUT"

prepare=$(sbatch --parsable --account="$ACCOUNT" --job-name=p37-overlap-data \
  --time=00:30:00 --cpus-per-task=2 --mem=16G \
  --output="$LOG_DIR/prepare-%j.log" --export="ALL,$common" "$SCRIPT_DIR/run_prepare.sh")

eval_jobs=()
for scale in 10000 100000; do
  for arm in joint denovo; do
    job=$(sbatch --parsable --account="$ACCOUNT" --job-name="p37-e-$((scale / 1000))k-${arm:0:1}" \
      --time=01:30:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
      --dependency="afterok:$prepare" --output="$LOG_DIR/eval-$scale-$arm-%j.log" \
      --export="ALL,$common,P37_SCALE=$scale,P37_ARM=$arm" "$SCRIPT_DIR/run_eval.sh")
    eval_jobs+=("$job")
  done
done
dependency=$(IFS=:; echo "${eval_jobs[*]}")
collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=p37-overlap-collect \
  --time=00:20:00 --cpus-per-task=2 --mem=8G --dependency="afterok:$dependency" \
  --output="$LOG_DIR/collect-%j.log" --export="ALL,$common" "$SCRIPT_DIR/run_collect.sh")
printf 'prepare=%s eval_jobs=%s collect=%s output=%s\n' \
  "$prepare" "${eval_jobs[*]}" "$collect" "$OUT"
