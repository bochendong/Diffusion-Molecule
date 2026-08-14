#!/usr/bin/env bash
# Submit B10 matched pilot and a CPU controller for a gated scale-up.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_VALENCE_HIER_VQ_SEED:-1741}"
LOG_DIR="${SUCC_VALENCE_HIER_VQ_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/valence_budget_hierarchical_vq_graph_flow_v10}"
OUTPUT_DIR="${SUCC_VALENCE_HIER_VQ_OUTPUT_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/valence_budget_hierarchical_vq_graph_flow_v10/seed_${SEED}}"
MAIL_USER="${SUCC_VALENCE_HIER_VQ_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"

pilot_submission="$(sbatch \
  --job-name="uca-valence-hvq-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_VALENCE_HIER_VQ_TIME:-00:30:00}" \
  --cpus-per-task="${SUCC_VALENCE_HIER_VQ_CPUS:-4}" \
  --mem="${SUCC_VALENCE_HIER_VQ_MEM:-16G}" \
  --gres="${SUCC_VALENCE_HIER_VQ_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-valence-hvq-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_valence_budget_hierarchical_vq_graph_flow_pilot.sh'")"
pilot_job_id="$(printf '%s\n' "$pilot_submission" | awk '{print $NF}')"

controller_submission="$(sbatch \
  --job-name="uca-valence-hvq-gate" \
  --account=def-hup-ab \
  --time=00:05:00 \
  --cpus-per-task=1 \
  --mem=1G \
  --dependency="afterok:${pilot_job_id}" \
  --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-valence-hvq-gate-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_valence_budget_scale_controller.sh' '$OUTPUT_DIR/summary.json'")"
controller_job_id="$(printf '%s\n' "$controller_submission" | awk '{print $NF}')"

echo "$pilot_submission"
echo "$controller_submission"
echo "valence_budget_pilot_job=$pilot_job_id"
echo "valence_budget_scale_controller_job=$controller_job_id"
echo "pilot_summary=$OUTPUT_DIR/summary.json"
