#!/usr/bin/env bash
# Submit the minimal CPU -> H100 20GB MIG -> CPU v11 signal DAG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_ANCHOR_RESIDUAL_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_anchor_residual_v11/seed_1714}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_anchor_residual_v11/seed_1714}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
CPU_ACCOUNT="${SUCC_UCA_CPU_ACCOUNT:-def-hup-ab_cpu}"
GPU_ACCOUNT="${SUCC_UCA_GPU_ACCOUNT:-def-hup-ab_gpu}"
GPU_REQUEST="${SUCC_UCA_GPU_REQUEST:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
mkdir -p "$LOG_DIR"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_ANCHOR_RESIDUAL_ROOT="$RUN_ROOT"

submit() {
  local output
  output="$(sbatch --parsable "$@")"
  printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1
}

prepare_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-anchor-r11-prep \
  --time=00:20:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_anchor_residual_v11.sh' prepare")"

gpu_job="$(submit --account="$GPU_ACCOUNT" --job-name=uca-anchor-r11-1p5b \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --time=02:30:00 --cpus-per-task=4 --mem=32G --gpus="$GPU_REQUEST" \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_anchor_residual_v11.sh' gpu")"

gate_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-anchor-r11-gate \
  --dependency="afterok:$gpu_job" --kill-on-invalid-dep=yes \
  --time=01:00:00 --cpus-per-task=8 --mem=32G \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_anchor_residual_v11.sh' oracle_gate")"

cat <<EOF
anchor_residual_v11_submitted
prepare_job=$prepare_job
gpu_job=$gpu_job
gate_job=$gate_job
candidate_budget=20
table1_anchor_top_k=5
evaluation_target_access=false
requested_accelerator_hours=2.5
output=$RUN_ROOT/gate/summary.json
EOF
