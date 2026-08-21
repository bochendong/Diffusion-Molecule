#!/usr/bin/env bash
# Submit paired H100 freeze array -> CPU evaluation array -> science gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_V8_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/paired_signed_set_calibration_v8}"
MAIL_USER="${SUCC_V8_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_V8_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_V8_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

freeze_job="$(sbatch --parsable \
  --job-name=uca-v8-paired-freeze \
  --account="$ACCOUNT" --time=01:30:00 --cpus-per-task=4 --mem=24G \
  --gres="gpu:$GPU_REQUEST" --array=0-2%2 \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v8-paired-freeze-%A_%a.log" \
  --wrap="SUCC_V8_STAGE=freeze bash '$SCRIPT_DIR/run_paired_signed_set_calibration_v8.sh'")"

evaluate_job="$(sbatch --parsable \
  --job-name=uca-v8-paired-eval \
  --account="$ACCOUNT" --time=00:30:00 --cpus-per-task=2 --mem=8G \
  --array=0-2%3 --dependency="afterok:$freeze_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-v8-paired-eval-%A_%a.log" \
  --wrap="SUCC_V8_STAGE=evaluate bash '$SCRIPT_DIR/run_paired_signed_set_calibration_v8.sh'")"

gate_job="$(sbatch --parsable \
  --job-name=uca-v8-paired-scigate \
  --account="$ACCOUNT" --time=00:10:00 --cpus-per-task=1 --mem=2G \
  --dependency="afterok:$evaluate_job" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-v8-paired-scigate-%j.log" \
  --wrap="SUCC_V8_STAGE=gate bash '$SCRIPT_DIR/run_paired_signed_set_calibration_v8.sh'")"

echo "freeze_job=$freeze_job"
echo "evaluate_job=$evaluate_job"
echo "gate_job=$gate_job"
echo "gpu=$GPU_REQUEST"
echo "candidate_contract=3_replicates_x_5_arms_x_48_conditions_x_exact_20"
echo "randomness_contract=paired_common_random_numbers_across_arms"
echo "science_contract=scientific_stop_exits_zero"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/paired_signed_set_calibration_v8/seed_2111/gate/gate_summary.json"
