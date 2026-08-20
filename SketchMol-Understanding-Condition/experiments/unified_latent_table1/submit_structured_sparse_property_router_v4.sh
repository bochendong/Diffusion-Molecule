#!/usr/bin/env bash
# Submit four execution arms, then a distinct scientific-gate job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_SPARSE_ROUTER_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/structured_sparse_property_router_v4}"
MAIL_USER="${SUCC_SPARSE_ROUTER_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_SPARSE_ROUTER_ACCOUNT:-def-hup-ab}"
GPU_REQUEST="${SUCC_SPARSE_ROUTER_GPU:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
mkdir -p "$LOG_DIR"

execution_job_id="$(sbatch --parsable \
  --job-name=uca-sparse-v4-exec \
  --account="$ACCOUNT" --time=00:30:00 --cpus-per-task=4 --mem=24G \
  --gres="gpu:$GPU_REQUEST" --array=0-3%2 \
  --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/uca-sparse-v4-exec-%A_%a.log" \
  --wrap="bash '$SCRIPT_DIR/run_structured_sparse_property_router_v4.sh'")"

gate_job_id="$(sbatch --parsable \
  --job-name=uca-sparse-v4-scigate \
  --account="$ACCOUNT" --time=00:05:00 --cpus-per-task=1 --mem=2G \
  --dependency="afterok:$execution_job_id" --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-sparse-v4-scigate-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_structured_sparse_property_router_v4_gate.sh'")"

echo "execution_job_id=$execution_job_id"
echo "gate_job_id=$gate_job_id"
echo "execution_contract=all_valid_arm_runs_exit_zero"
echo "science_gate_contract=only_gate_job_exits_nonzero_on_scientific_stop"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/structured_sparse_property_router_v4/seed_2061/gate/gate_summary.json"
