#!/usr/bin/env bash
# Submit verifier-aligned preference v2 on one H100 20 GB MIG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SEED="${SUCC_UCA_SEED:-1705}"
ACCOUNT="${SUCC_UCA_SLURM_ACCOUNT:-rrg-hup}"
GPU="${SUCC_UCA_SLURM_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-/scratch/bdong/projects/Diffusion-Molecule/SketchMol-Understanding-Condition/logs/unified_constraint_agent_verifier_preference_v2}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name="uca-vpref-s${SEED}" \
  --time="${SUCC_UCA_VERIFIER_PREFERENCE_TIME:-01:30:00}" \
  --cpus-per-task="${SUCC_UCA_VERIFIER_PREFERENCE_CPUS:-8}" \
  --mem="${SUCC_UCA_VERIFIER_PREFERENCE_MEM:-64G}" \
  --gpus="$GPU" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_common_llm_verifier_preference_pilot.sh'")"
job_id="${submission%%;*}"
echo "Common-LLM verifier preference v2 submitted"
echo "  job_id=$job_id"
echo "  log=$LOG_DIR/uca-vpref-s${SEED}-$job_id.log"
