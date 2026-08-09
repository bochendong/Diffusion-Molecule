#!/usr/bin/env bash
# Submit base-vs-tuned common-LLM held-out evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SEED="${SUCC_UCA_SEED:-1703}"
ACCOUNT="${SUCC_UCA_SLURM_ACCOUNT:-rrg-hup}"
GPU="${SUCC_UCA_SLURM_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
DEPENDENCY="${SUCC_UCA_SLURM_DEPENDENCY:-}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/unified_constraint_agent_common_llm_pilot_v1}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
mkdir -p "$LOG_DIR"

args=(
  --parsable
  --account="$ACCOUNT"
  --job-name="uca-llm-eval-s${SEED}"
  --time="${SUCC_UCA_EVAL_TIME:-01:00:00}"
  --cpus-per-task="${SUCC_UCA_EVAL_CPUS:-8}"
  --mem="${SUCC_UCA_EVAL_MEM:-64G}"
  --gpus="$GPU"
  --mail-user="$MAIL_USER"
  --mail-type=BEGIN,END,FAIL
  --output="$LOG_DIR/%x-%j.log"
  --export=ALL
)
[[ -n "$DEPENDENCY" ]] && args+=(--dependency="$DEPENDENCY")

submission="$(sbatch "${args[@]}" --wrap="bash '$SCRIPT_DIR/run_common_llm_eval.sh'")"
job_id="${submission%%;*}"
echo "Common-LLM held-out evaluation submitted"
echo "  job_id=$job_id"
echo "  dependency=${DEPENDENCY:-none}"
echo "  log=$LOG_DIR/uca-llm-eval-s${SEED}-$job_id.log"
