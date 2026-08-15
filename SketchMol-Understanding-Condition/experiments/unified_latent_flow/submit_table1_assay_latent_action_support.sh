#!/usr/bin/env bash
# Submit eight parallel B30-r1 CPU shards and a dependent merge.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_ASSAY_SUPPORT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/table1_assay_latent_action_support_v30_r1}"
MAIL_USER="${SUCC_ASSAY_SUPPORT_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
array_submission="$(sbatch \
  --job-name="uca-assay-v30r1" \
  --account=def-hup-ab_cpu \
  --array="0-7%8" \
  --time="00:45:00" \
  --cpus-per-task=1 \
  --mem=6G \
  --mail-user="$MAIL_USER" \
  --mail-type=FAIL \
  --output="$LOG_DIR/uca-assay-v30r1-%A_%a.log" \
  --wrap="bash '$SCRIPT_DIR/run_table1_assay_latent_action_support.sh'")"
array_job="$(printf '%s\n' "$array_submission" | awk '{print $NF}')"

merge_submission="$(sbatch \
  --job-name="uca-assay-merge-v30r1" \
  --account=def-hup-ab_cpu \
  --dependency="afterok:$array_job" \
  --kill-on-invalid-dep=yes \
  --time="00:05:00" \
  --cpus-per-task=1 \
  --mem=2G \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-assay-merge-v30r1-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_merge_table1_assay_latent_action_support.sh'")"
merge_job="$(printf '%s\n' "$merge_submission" | awk '{print $NF}')"

echo "$array_submission"
echo "$merge_submission"
echo "assay_support_array_job=$array_job"
echo "assay_support_merge_job=$merge_job"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/table1_assay_latent_action_support_v30_r1/seed_1922/merged/summary.json"
