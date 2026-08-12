#!/usr/bin/env bash
# Submit the fast CPU-array -> H100 20GB MIG -> CPU oracle v9 DAG on Nibi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_MUMO_RESIDUAL_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_mumo_residual_planner_v9/seed_1712}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_mumo_residual_planner_v9/seed_1712}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
CPU_ACCOUNT="${SUCC_UCA_CPU_ACCOUNT:-def-hup-ab_cpu}"
GPU_ACCOUNT="${SUCC_UCA_GPU_ACCOUNT:-def-hup-ab_gpu}"
GPU_REQUEST="${SUCC_UCA_GPU_REQUEST:-nvidia_h100_80gb_hbm3_2g.20gb:1}"
SHARD_COUNT="${SUCC_UCA_MUMO_DEV_SHARD_COUNT:-16}"
SEED="${SUCC_UCA_SEED:-1712}"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
mkdir -p "$LOG_DIR"
parse_id() { printf '%s\n' "$1" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1; }
submit_and_parse() {
  local output
  output="$(sbatch "$@")"
  local job_id
  job_id="$(parse_id "$output")"
  [[ -n "$job_id" ]] || { echo "ERROR: could not parse Slurm job id from: $output" >&2; exit 2; }
  printf '%s\n' "$job_id"
}

export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_MUMO_RESIDUAL_ROOT="$RUN_ROOT"
export SUCC_UCA_SEED="$SEED"

prepare_id="$(submit_and_parse --parsable --account="$CPU_ACCOUNT" \
  --job-name=uca-mumo-r9-prepare --time=00:30:00 --cpus-per-task=4 --mem=16G \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_mumo_residual_planner_v9.sh' prepare")"

array_end="$((SHARD_COUNT - 1))"
ARRAY_SPEC="${SUCC_UCA_MUMO_RESIDUAL_ARRAY_SPEC:-0-${array_end}%${SHARD_COUNT}}"
[[ "$ARRAY_SPEC" =~ ^[0-9,%\-]+$ ]] || { echo "ERROR: invalid array spec" >&2; exit 2; }
enumerate_id="$(submit_and_parse --parsable --account="$CPU_ACCOUNT" \
  --job-name=uca-mumo-r9-enum --array="$ARRAY_SPEC" --time=02:00:00 \
  --cpus-per-task=4 --mem=12G --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/%x-%A_%a.log" --export=ALL,SUCC_UCA_MUMO_DEV_SHARD_COUNT="$SHARD_COUNT" \
  --wrap="bash '$SCRIPT_DIR/run_mumo_residual_planner_v9.sh' enumerate")"

merge_id="$(submit_and_parse --parsable --account="$CPU_ACCOUNT" \
  --job-name=uca-mumo-r9-merge --dependency="afterok:$enumerate_id" --kill-on-invalid-dep=yes \
  --time=00:20:00 --cpus-per-task=2 --mem=8G --mail-user="$MAIL_USER" --mail-type=FAIL \
  --output="$LOG_DIR/%x-%j.log" --export=ALL,SUCC_UCA_MUMO_DEV_SHARD_COUNT="$SHARD_COUNT" \
  --wrap="bash '$SCRIPT_DIR/run_mumo_residual_planner_v9.sh' merge")"

gpu_id="$(submit_and_parse --parsable --account="$GPU_ACCOUNT" \
  --job-name=uca-mumo-r9-1p5b --dependency="afterok:$prepare_id:$merge_id" --kill-on-invalid-dep=yes \
  --time=02:00:00 --cpus-per-task=4 --mem=32G --gpus="$GPU_REQUEST" \
  --mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_mumo_residual_planner_v9.sh' gpu")"

gate_id="$(submit_and_parse --parsable --account="$CPU_ACCOUNT" \
  --job-name=uca-mumo-r9-gate --dependency="afterok:$gpu_id" --kill-on-invalid-dep=yes \
  --time=04:00:00 --cpus-per-task=8 --mem=32G --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL --output="$LOG_DIR/%x-%j.log" --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_mumo_residual_planner_v9.sh' oracle_gate")"

echo "prepare_job=$prepare_id"
echo "enumerate_array_job=$enumerate_id"
echo "enumerate_array_spec=$ARRAY_SPEC"
echo "merge_job=$merge_id"
echo "gpu_job=$gpu_id"
echo "oracle_gate_job=$gate_id"
echo "candidate_budget=20"
echo "evaluation_target_access=false"
echo "gpu=$GPU_REQUEST"
echo "requested_accelerator_hours=2.0"
echo "output=$RUN_ROOT"
