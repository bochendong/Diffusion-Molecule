#!/usr/bin/env bash
# Submit Table1 and MuMO official common-LLM edit evaluations in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

ACCOUNT="${SUCC_UCA_SLURM_ACCOUNT:-rrg-hup}"
GPU="${SUCC_UCA_SLURM_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/unified_constraint_agent_official_edit_v1}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
mkdir -p "$LOG_DIR"

submit_suite() {
  local suite="$1"
  local time_limit="$2"
  local cpus="$3"
  local memory="$4"
  local submission job_id
  submission="$(sbatch \
    --parsable \
    --account="$ACCOUNT" \
    --job-name="uca-official-${suite}" \
    --time="$time_limit" \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --gpus="$GPU" \
    --mail-user="$MAIL_USER" \
    --mail-type=BEGIN,END,FAIL \
    --output="$LOG_DIR/%x-%j.log" \
    --export=ALL \
    --wrap="bash '$SCRIPT_DIR/run_common_llm_official_edit_eval.sh' '$suite'")"
  job_id="${submission%%;*}"
  echo "$suite job_id=$job_id log=$LOG_DIR/uca-official-${suite}-$job_id.log"
}

echo "Submitting official common-LLM edit evaluations"
submit_suite table1 "${SUCC_UCA_TABLE1_TIME:-03:00:00}" "${SUCC_UCA_TABLE1_CPUS:-8}" "${SUCC_UCA_TABLE1_MEM:-64G}"
submit_suite mumo "${SUCC_UCA_MUMO_TIME:-03:00:00}" "${SUCC_UCA_MUMO_CPUS:-12}" "${SUCC_UCA_MUMO_MEM:-96G}"
