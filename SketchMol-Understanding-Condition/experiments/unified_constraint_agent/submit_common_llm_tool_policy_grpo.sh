#!/usr/bin/env bash
# Submit the fast 1.5B closed-loop tool-policy pilot on one H100 20 GB MIG slice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SEED="${SUCC_UCA_SEED:-1707}"
ACCOUNT="${SUCC_UCA_SLURM_ACCOUNT:-rrg-hup}"
GPU="${SUCC_UCA_SLURM_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/unified_constraint_agent_tool_policy_grpo_v1}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
RUNNER="$SCRIPT_DIR/run_common_llm_tool_policy_grpo.sh"
mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name="uca-tool-grpo-s${SEED}" \
  --time="${SUCC_UCA_TIME:-04:00:00}" \
  --cpus-per-task="${SUCC_UCA_CPUS:-4}" \
  --mem="${SUCC_UCA_MEM:-80G}" \
  --gpus="$GPU" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --wrap="bash '$RUNNER'")"
job_id="${submission%%;*}"

echo "Common-LLM tool-policy GRPO pilot submitted"
echo "  job_id=$job_id"
echo "  seed=$SEED"
echo "  gpu=$GPU"
echo "  mail=$MAIL_USER"
echo "  log=$LOG_DIR/uca-tool-grpo-s${SEED}-$job_id.log"
