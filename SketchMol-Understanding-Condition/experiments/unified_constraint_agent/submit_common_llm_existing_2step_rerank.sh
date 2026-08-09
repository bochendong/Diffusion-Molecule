#!/usr/bin/env bash
# Submit common-LLM reranking of the existing official MuMO 2-step top-20 pool.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

ACCOUNT="${SUCC_UCA_SLURM_ACCOUNT:-rrg-hup}"
GPU="${SUCC_UCA_SLURM_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/unified_constraint_agent_existing_2step_rerank_v1}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name="uca-rerank-2step" \
  --time="${SUCC_UCA_2STEP_TIME:-03:00:00}" \
  --cpus-per-task="${SUCC_UCA_2STEP_CPUS:-8}" \
  --mem="${SUCC_UCA_2STEP_MEM:-64G}" \
  --gpus="$GPU" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_common_llm_existing_2step_rerank.sh'")"
job_id="${submission%%;*}"
echo "Common-LLM existing 2-step rerank submitted"
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/uca-rerank-2step-$job_id.log"
