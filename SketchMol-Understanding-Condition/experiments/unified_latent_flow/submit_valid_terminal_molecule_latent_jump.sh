#!/usr/bin/env bash
# Submit one bounded valid-terminal graph-latent run on a 10GB H100 MIG slice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_VALID_TERMINAL_SEED:-1991}"
LOG_DIR="${SUCC_VALID_TERMINAL_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/valid_terminal_molecule_latent_jump_v1}"
MAIL_USER="${SUCC_VALID_TERMINAL_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-valid-state-v1" \
  --account=def-hup-ab \
  --time="${SUCC_VALID_TERMINAL_TIME:-00:45:00}" \
  --cpus-per-task="${SUCC_VALID_TERMINAL_CPUS:-4}" \
  --mem="${SUCC_VALID_TERMINAL_MEM:-20G}" \
  --gres="${SUCC_VALID_TERMINAL_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-valid-state-v1-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_valid_terminal_molecule_latent_jump.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "valid_terminal_job=$job_id"
echo "log=$LOG_DIR/uca-valid-state-v1-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/valid_terminal_molecule_latent_jump_v1/seed_${SEED}/summary.json"
