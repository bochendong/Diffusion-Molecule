#!/usr/bin/env bash
# Submit CPU prepare -> matched CPU/GPU controllers -> oracle -> v14a gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_FEEDBACK_REPAIR_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_feedback_repair_v14a/seed_1716}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_feedback_repair_v14a/seed_1716}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
CPU_ACCOUNT="${SUCC_UCA_CPU_ACCOUNT:-def-hup-ab_cpu}"
GPU_ACCOUNT="${SUCC_UCA_GPU_ACCOUNT:-def-hup-ab_gpu}"
GPU_REQUEST="${SUCC_UCA_GPU_REQUEST:-nvidia_h100_80gb_hbm3_2g.20gb:1}"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
mkdir -p "$LOG_DIR"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_FEEDBACK_REPAIR_ROOT="$RUN_ROOT"
submit() {
  local output
  output="$(sbatch --parsable "$@")"
  printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1
}

prepare_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-prep \
  --time=00:45:00 --cpus-per-task=4 --mem=24G \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' prepare")"
gpu_job="$(submit --account="$GPU_ACCOUNT" --job-name=uca-feedback-v14a-llm \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --time=03:30:00 --cpus-per-task=4 --mem=32G --gpus="$GPU_REQUEST" \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' gpu")"
deterministic_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-det \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --time=01:30:00 --cpus-per-task=8 --mem=32G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' deterministic")"
llm_oracle_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-llo \
  --dependency="afterok:$gpu_job" --kill-on-invalid-dep=yes \
  --time=02:00:00 --cpus-per-task=8 --mem=32G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' oracle_llm")"
det_oracle_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-deo \
  --dependency="afterok:$deterministic_job" --kill-on-invalid-dep=yes \
  --time=02:00:00 --cpus-per-task=8 --mem=32G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' oracle_deterministic")"
gate_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-feedback-v14a-gate \
  --dependency="afterok:$llm_oracle_job:$det_oracle_job" --kill-on-invalid-dep=yes \
  --time=00:20:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_feedback_repair_v14a.sh' gate")"

cat <<EOF
feedback_repair_v14a_submitted
prepare_job=$prepare_job
llm_gpu_job=$gpu_job
deterministic_job=$deterministic_job
llm_oracle_job=$llm_oracle_job
deterministic_oracle_job=$det_oracle_job
gate_job=$gate_job
signal_conditions=200
generation_attempts_per_condition=20
output_selection=none
evaluation_target_access=false
requested_accelerator_hours=3.5
output=$RUN_ROOT/gate/summary.json
EOF
