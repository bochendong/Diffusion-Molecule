#!/usr/bin/env bash
# Submit CPU preparation followed by GPU training/evaluation for faster queueing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SEED="${SUCC_UCA_SEED:-1706}"
CPU_ACCOUNT="${SUCC_UCA_CPU_ACCOUNT:-def-hup-ab}"
GPU_ACCOUNT="${SUCC_UCA_SLURM_ACCOUNT:-rrg-hup}"
GPU="${SUCC_UCA_SLURM_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/unified_constraint_agent_two_step_plan_preference_v3}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
RUNNER="$SCRIPT_DIR/run_common_llm_two_step_plan_preference.sh"
mkdir -p "$LOG_DIR"

prepare_submission="$(sbatch \
  --parsable \
  --account="$CPU_ACCOUNT" \
  --job-name="uca-plan-v3-prep-s${SEED}" \
  --time="${SUCC_UCA_PREP_TIME:-06:00:00}" \
  --cpus-per-task="${SUCC_UCA_PREP_CPUS:-4}" \
  --mem="${SUCC_UCA_PREP_MEM:-64G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL,SUCC_UCA_STAGE=prepare \
  --wrap="bash '$RUNNER'")"
prepare_job_id="${prepare_submission%%;*}"

gpu_submission="$(sbatch \
  --parsable \
  --account="$GPU_ACCOUNT" \
  --job-name="uca-plan-v3-fit-s${SEED}" \
  --dependency="afterok:$prepare_job_id" \
  --kill-on-invalid-dep=yes \
  --time="${SUCC_UCA_FIT_TIME:-06:00:00}" \
  --cpus-per-task="${SUCC_UCA_PLAN_CPUS:-4}" \
  --mem="${SUCC_UCA_PLAN_MEM:-96G}" \
  --gpus="$GPU" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL,SUCC_UCA_STAGE=train_eval \
  --wrap="bash '$RUNNER'")"
gpu_job_id="${gpu_submission%%;*}"

echo "Common-LLM two-step plan preference v3 split pipeline submitted"
echo "  prepare_job_id=$prepare_job_id"
echo "  gpu_job_id=$gpu_job_id"
echo "  dependency=afterok:$prepare_job_id"
echo "  seed=$SEED"
echo "  gpu=$GPU"
echo "  mail=$MAIL_USER"
echo "  prepare_log=$LOG_DIR/uca-plan-v3-prep-s${SEED}-$prepare_job_id.log"
echo "  gpu_log=$LOG_DIR/uca-plan-v3-fit-s${SEED}-$gpu_job_id.log"
