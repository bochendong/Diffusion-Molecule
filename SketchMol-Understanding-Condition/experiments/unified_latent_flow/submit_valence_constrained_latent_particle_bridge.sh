#!/usr/bin/env bash
# Submit one bounded B40 frozen-checkpoint run on a 10GB H100 MIG slice.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_VALENCE_PARTICLE_SEED:-1989}"
LOG_DIR="${SUCC_VALENCE_PARTICLE_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/valence_constrained_latent_particle_bridge_v40}"
MAIL_USER="${SUCC_VALENCE_PARTICLE_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
submission="$(sbatch \
  --job-name="uca-val-particle-v40" \
  --account=def-hup-ab \
  --time="${SUCC_VALENCE_PARTICLE_TIME:-00:30:00}" \
  --cpus-per-task="${SUCC_VALENCE_PARTICLE_CPUS:-4}" \
  --mem="${SUCC_VALENCE_PARTICLE_MEM:-20G}" \
  --gres="${SUCC_VALENCE_PARTICLE_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-val-particle-v40-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_valence_constrained_latent_particle_bridge.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "valence_particle_bridge_job=$job_id"
echo "log=$LOG_DIR/uca-val-particle-v40-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/valence_constrained_latent_particle_bridge_v40/seed_${SEED}/summary.json"
