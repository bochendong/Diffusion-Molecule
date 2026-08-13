#!/usr/bin/env bash
# Submit CPU prepare -> 20GB MIG controller -> CPU trajectory/oracle v12 DAG.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_DIRECT_REPAIR_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_direct_repair_v12/seed_1715}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_direct_repair_v12/seed_1715}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
CPU_ACCOUNT="${SUCC_UCA_CPU_ACCOUNT:-def-hup-ab_cpu}"
GPU_ACCOUNT="${SUCC_UCA_GPU_ACCOUNT:-def-hup-ab_gpu}"
GPU_REQUEST="${SUCC_UCA_GPU_REQUEST:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
SHARD_COUNT="${SUCC_UCA_DIRECT_REPAIR_SHARD_COUNT:-16}"
command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
mkdir -p "$LOG_DIR"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_DIRECT_REPAIR_ROOT="$RUN_ROOT"

submit() {
  local output
  output="$(sbatch --parsable "$@")"
  printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1
}

prepare_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-direct-r12-prep \
  --time=00:30:00 --cpus-per-task=4 --mem=16G \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_direct_repair_v12.sh' prepare")"

gpu_job="$(submit --account="$GPU_ACCOUNT" --job-name=uca-direct-r12-ctrl \
  --dependency="afterok:$prepare_job" --kill-on-invalid-dep=yes \
  --time=02:00:00 --cpus-per-task=4 --mem=32G --gpus="$GPU_REQUEST" \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_direct_repair_v12.sh' gpu")"

array_end="$((SHARD_COUNT - 1))"
trajectory_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-direct-r12-traj \
  --dependency="afterok:$gpu_job" --kill-on-invalid-dep=yes \
  --array="0-${array_end}%${SHARD_COUNT}" \
  --time=02:00:00 --cpus-per-task=8 --mem=24G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%A_%a.log" \
  --export=ALL,SUCC_UCA_DIRECT_REPAIR_SHARD_COUNT="$SHARD_COUNT" \
  --wrap="bash '$SCRIPT_DIR/run_direct_repair_v12.sh' trajectory")"

merge_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-direct-r12-merge \
  --dependency="afterok:$trajectory_job" --kill-on-invalid-dep=yes \
  --time=00:20:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL,SUCC_UCA_DIRECT_REPAIR_SHARD_COUNT="$SHARD_COUNT" \
  --wrap="bash '$SCRIPT_DIR/run_direct_repair_v12.sh' merge")"

oracle_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-direct-r12-oracle \
  --dependency="afterok:$merge_job" --kill-on-invalid-dep=yes \
  --time=03:00:00 --cpus-per-task=8 --mem=32G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_direct_repair_v12.sh' oracle")"

gate_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-direct-r12-gate \
  --dependency="afterok:$oracle_job" --kill-on-invalid-dep=yes \
  --time=00:20:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_direct_repair_v12.sh' gate")"

cat <<EOF
direct_repair_v12_submitted
prepare_job=$prepare_job
controller_gpu_job=$gpu_job
trajectory_array_job=$trajectory_job
merge_job=$merge_job
oracle_job=$oracle_job
gate_job=$gate_job
generation_attempts_per_condition=20
output_selection=none
evaluation_target_access=false
requested_accelerator_hours=2.0
output=$RUN_ROOT/gate/summary.json
EOF
