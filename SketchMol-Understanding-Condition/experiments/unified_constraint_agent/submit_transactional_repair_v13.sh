#!/usr/bin/env bash
# Submit the CPU-only transactional v13 signal using frozen v12 LLM plans.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
V12_ROOT="${SUCC_UCA_V12_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_direct_repair_v12/seed_1715}"
RUN_ROOT="${SUCC_UCA_TRANSACTIONAL_REPAIR_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_transactional_repair_v13/seed_1715}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_transactional_repair_v13/seed_1715}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
CPU_ACCOUNT="${SUCC_UCA_CPU_ACCOUNT:-def-hup-ab_cpu}"
SHARD_COUNT="${SUCC_UCA_TRANSACTIONAL_REPAIR_SHARD_COUNT:-16}"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }
[[ -s "$V12_ROOT/controller/dev_plans.jsonl" && -s "$V12_ROOT/controller/dev_plans.manifest.json" ]] || {
  echo "ERROR: frozen v12 common-LLM plans are unavailable" >&2; exit 2;
}
mkdir -p "$LOG_DIR"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_TRANSACTIONAL_REPAIR_ROOT="$RUN_ROOT"

submit() {
  local output
  output="$(sbatch --parsable "$@")"
  printf '%s\n' "$output" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1
}

array_end="$((SHARD_COUNT - 1))"
trajectory_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-trans-r13-traj \
  --array="0-${array_end}%${SHARD_COUNT}" \
  --time=02:00:00 --cpus-per-task=4 --mem=16G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%A_%a.log" \
  --export=ALL,SUCC_UCA_TRANSACTIONAL_REPAIR_SHARD_COUNT="$SHARD_COUNT" \
  --wrap="bash '$SCRIPT_DIR/run_transactional_repair_v13.sh' trajectory")"

merge_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-trans-r13-merge \
  --dependency="afterok:$trajectory_job" --kill-on-invalid-dep=yes \
  --time=00:20:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL,SUCC_UCA_TRANSACTIONAL_REPAIR_SHARD_COUNT="$SHARD_COUNT" \
  --wrap="bash '$SCRIPT_DIR/run_transactional_repair_v13.sh' merge")"

oracle_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-trans-r13-oracle \
  --dependency="afterok:$merge_job" --kill-on-invalid-dep=yes \
  --time=03:00:00 --cpus-per-task=8 --mem=32G \
  --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_transactional_repair_v13.sh' oracle")"

gate_job="$(submit --account="$CPU_ACCOUNT" --job-name=uca-trans-r13-gate \
  --dependency="afterok:$oracle_job" --kill-on-invalid-dep=yes \
  --time=00:20:00 --cpus-per-task=2 --mem=8G \
  --mail-user="$MAIL_USER" --mail-type=END,FAIL --output="$LOG_DIR/%x-%j.log" \
  --export=ALL --wrap="bash '$SCRIPT_DIR/run_transactional_repair_v13.sh' gate")"

cat <<EOF
transactional_repair_v13_submitted
trajectory_array_job=$trajectory_job
merge_job=$merge_job
oracle_job=$oracle_job
gate_job=$gate_job
generation_attempts_per_condition=20
output_selection=none
evaluation_target_access=false
requested_accelerator_hours=0
reused_frozen_common_llm_plans=$V12_ROOT/controller/dev_plans.jsonl
output=$RUN_ROOT/gate/summary.json
EOF
