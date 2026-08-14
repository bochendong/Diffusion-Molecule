#!/usr/bin/env bash
# Submit B11 matched pilot and a dependent scientific gate controller.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_MOTIF_ATTACH_SEED:-1741}"
LOG_DIR="${SUCC_MOTIF_ATTACH_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/motif_attachment_hierarchical_vq_graph_flow_v11}"
OUTPUT_DIR="${SUCC_MOTIF_ATTACH_OUTPUT_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/motif_attachment_hierarchical_vq_graph_flow_v11/seed_${SEED}}"
MAIL_USER="${SUCC_MOTIF_ATTACH_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"
pilot_submission="$(sbatch \
  --job-name="uca-motif-hvq-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_MOTIF_ATTACH_TIME:-00:30:00}" \
  --cpus-per-task="${SUCC_MOTIF_ATTACH_CPUS:-4}" \
  --mem="${SUCC_MOTIF_ATTACH_MEM:-16G}" \
  --gres="${SUCC_MOTIF_ATTACH_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-motif-hvq-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_motif_attachment_hierarchical_vq_graph_flow_pilot.sh'")"
pilot_job_id="$(printf '%s\n' "$pilot_submission" | awk '{print $NF}')"

controller_submission="$(sbatch \
  --job-name="uca-motif-hvq-gate" \
  --account=def-hup-ab \
  --time=00:05:00 \
  --cpus-per-task=1 \
  --mem=1G \
  --dependency="afterok:${pilot_job_id}" \
  --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-motif-hvq-gate-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_motif_attachment_scale_controller.sh' '$OUTPUT_DIR/summary.json'")"
controller_job_id="$(printf '%s\n' "$controller_submission" | awk '{print $NF}')"

echo "$pilot_submission"
echo "$controller_submission"
echo "motif_attachment_pilot_job=$pilot_job_id"
echo "motif_attachment_scale_controller_job=$controller_job_id"
echo "pilot_summary=$OUTPUT_DIR/summary.json"
