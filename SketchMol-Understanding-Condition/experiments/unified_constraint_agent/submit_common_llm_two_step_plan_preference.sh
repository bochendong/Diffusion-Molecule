#!/usr/bin/env bash
# Submit the leakage-safe two-step plan preference pipeline on one H100 20 GB MIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SEED="${SUCC_UCA_SEED:-1706}"
ACCOUNT="${SUCC_UCA_SLURM_ACCOUNT:-rrg-hup}"
GPU="${SUCC_UCA_SLURM_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/unified_constraint_agent_two_step_plan_preference_v3}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name="uca-plan-v3-s${SEED}" \
  --time="${SUCC_UCA_PLAN_TIME:-10:00:00}" \
  --cpus-per-task="${SUCC_UCA_PLAN_CPUS:-8}" \
  --mem="${SUCC_UCA_PLAN_MEM:-96G}" \
  --gpus="$GPU" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_common_llm_two_step_plan_preference.sh'")"
job_id="${submission%%;*}"
echo "Common-LLM two-step plan preference v3 submitted"
echo "  job_id=$job_id"
echo "  seed=$SEED"
echo "  gpu=$GPU"
echo "  mail=$MAIL_USER"
echo "  log=$LOG_DIR/uca-plan-v3-s${SEED}-$job_id.log"
