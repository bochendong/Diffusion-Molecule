#!/usr/bin/env bash
# Submit one bounded B41 run on a 10GB H100 MIG slice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_VIABILITY_PARTICLE_SEED:-1991}"
LOG_DIR="${SUCC_VIABILITY_PARTICLE_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/viability_preserving_interacting_particle_transport_v41}"
MAIL_USER="${SUCC_VIABILITY_PARTICLE_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-viable-particle-v41" \
  --account=def-hup-ab \
  --time="${SUCC_VIABILITY_PARTICLE_TIME:-00:45:00}" \
  --cpus-per-task="${SUCC_VIABILITY_PARTICLE_CPUS:-4}" \
  --mem="${SUCC_VIABILITY_PARTICLE_MEM:-20G}" \
  --gres="${SUCC_VIABILITY_PARTICLE_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-viable-particle-v41-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_viability_preserving_interacting_particle_transport.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "viability_particle_transport_job=$job_id"
echo "log=$LOG_DIR/uca-viable-particle-v41-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/viability_preserving_interacting_particle_transport_v41/seed_${SEED}/summary.json"
