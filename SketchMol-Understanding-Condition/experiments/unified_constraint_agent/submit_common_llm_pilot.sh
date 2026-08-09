#!/usr/bin/env bash
# Submit the 1.5B common-LLM LoRA pilot to one H100 20 GB MIG by default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

ACCOUNT="${SUCC_UCA_SLURM_ACCOUNT:-rrg-hup}"
GPU="${SUCC_UCA_SLURM_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/unified_constraint_agent_common_llm_pilot_v1}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
DEPENDENCY="${SUCC_UCA_SLURM_DEPENDENCY:-}"
SEED="${SUCC_UCA_SEED:-1701}"
mkdir -p "$LOG_DIR"

args=(
  --parsable
  --account="$ACCOUNT"
  --job-name="${SUCC_UCA_JOB_NAME:-uca-llm-s${SEED}}"
  --time="${SUCC_UCA_TIME:-04:00:00}"
  --cpus-per-task="${SUCC_UCA_CPUS:-8}"
  --mem="${SUCC_UCA_MEM:-64G}"
  --gpus="$GPU"
  --mail-user="$MAIL_USER"
  --mail-type=BEGIN,END,FAIL
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL
)
[[ -n "$DEPENDENCY" ]] && args+=(--dependency="$DEPENDENCY")

submission="$(sbatch "${args[@]}" --wrap="bash '$SCRIPT_DIR/run_common_llm_pilot.sh'")"
job_id="${submission%%;*}"
echo "Common-LLM LoRA pilot submitted"
echo "  job_id=$job_id"
echo "  gpu=$GPU"
echo "  dependency=${DEPENDENCY:-none}"
echo "  mail=$MAIL_USER"
echo "  log=$LOG_DIR/uca-llm-s${SEED}-$job_id.log"
