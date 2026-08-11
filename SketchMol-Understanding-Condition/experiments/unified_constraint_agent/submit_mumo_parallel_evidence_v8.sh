#!/usr/bin/env bash
# Submit the CPU fan-out/fan-in DAG for MuMO train-only delta/verifier evidence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found" >&2; exit 2; }

SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_MUMO_PARALLEL_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_mumo_parallel_evidence_v8/seed_1711}"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_mumo_parallel_evidence_v8/seed_1711}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_UCA_ACCOUNT:-def-hup-ab_cpu}"
SHARD_COUNT="${SUCC_UCA_MUMO_SHARDS:-32}"
MAX_PARALLEL="${SUCC_UCA_MAX_PARALLEL_CPU_TASKS:-16}"

if [[ "$SHARD_COUNT" != "32" ]]; then
  echo "ERROR: the reviewed v8 submission contract fixes 32 shards" >&2
  exit 2
fi
mkdir -p "$LOG_DIR" "$RUN_ROOT"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_MUMO_PARALLEL_ROOT="$RUN_ROOT"
export SUCC_UCA_MUMO_SHARDS="$SHARD_COUNT"

parse_job_id() {
  printf '%s\n' "$1" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1
}

echo "Submitting MuMO v8 parallel evidence DAG"
echo "  data=MuMO train only"
echo "  balanced_sample=up to 5500 unique pairs/task (fit+dev; rare tasks keep all)"
echo "  candidate_budget=20 (locked for downstream gate)"
echo "  evaluation_target_access=false"
echo "  official_test_content_access=false (digest only)"
echo "  accelerators=none"
echo "  cpu_shards=32 max_parallel=$MAX_PARALLEL"

prepare_out="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name=uca-mumo-v8-prepare \
  --time="${SUCC_UCA_PREPARE_TIME:-00:45:00}" \
  --cpus-per-task=4 \
  --mem=8G \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_mumo_parallel_evidence_v8.sh' prepare")"
prepare_job="$(parse_job_id "$prepare_out")"
[[ -n "$prepare_job" ]] || { echo "ERROR: could not parse prepare job: $prepare_out" >&2; exit 2; }

delta_out="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name=uca-mumo-v8-delta \
  --array="0-31%${MAX_PARALLEL}" \
  --dependency="afterok:$prepare_job" \
  --kill-on-invalid-dep=yes \
  --time="${SUCC_UCA_DELTA_TIME:-01:00:00}" \
  --cpus-per-task=1 \
  --mem=6G \
  --mail-user="$MAIL_USER" \
  --mail-type=FAIL \
  --output="$LOG_DIR/%x-%A_%a.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_mumo_parallel_evidence_v8.sh' delta")"
delta_job="$(parse_job_id "$delta_out")"
[[ -n "$delta_job" ]] || { echo "ERROR: could not parse delta array: $delta_out" >&2; exit 2; }

feature_out="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name=uca-mumo-v8-features \
  --array="0-31%${MAX_PARALLEL}" \
  --dependency="afterok:$prepare_job" \
  --kill-on-invalid-dep=yes \
  --time="${SUCC_UCA_FEATURE_TIME:-01:00:00}" \
  --cpus-per-task=1 \
  --mem=8G \
  --mail-user="$MAIL_USER" \
  --mail-type=FAIL \
  --output="$LOG_DIR/%x-%A_%a.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_mumo_parallel_evidence_v8.sh' features")"
feature_job="$(parse_job_id "$feature_out")"
[[ -n "$feature_job" ]] || { echo "ERROR: could not parse feature array: $feature_out" >&2; exit 2; }

verifier_out="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name=uca-mumo-v8-verifier \
  --array="0-4%5" \
  --dependency="afterok:$feature_job" \
  --kill-on-invalid-dep=yes \
  --time="${SUCC_UCA_VERIFIER_TIME:-02:00:00}" \
  --cpus-per-task=8 \
  --mem=24G \
  --mail-user="$MAIL_USER" \
  --mail-type=FAIL \
  --output="$LOG_DIR/%x-%A_%a.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_mumo_parallel_evidence_v8.sh' verifier")"
verifier_job="$(parse_job_id "$verifier_out")"
[[ -n "$verifier_job" ]] || { echo "ERROR: could not parse verifier array: $verifier_out" >&2; exit 2; }

finalize_out="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name=uca-mumo-v8-gate \
  --dependency="afterok:$delta_job:$verifier_job" \
  --kill-on-invalid-dep=yes \
  --time="${SUCC_UCA_GATE_TIME:-00:30:00}" \
  --cpus-per-task=2 \
  --mem=16G \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_mumo_parallel_evidence_v8.sh' finalize")"
finalize_job="$(parse_job_id "$finalize_out")"
[[ -n "$finalize_job" ]] || { echo "ERROR: could not parse gate job: $finalize_out" >&2; exit 2; }

echo "prepare_job=$prepare_job"
echo "delta_array_job=$delta_job"
echo "feature_array_job=$feature_job"
echo "verifier_array_job=$verifier_job"
echo "evidence_gate_job=$finalize_job"
echo "output=$RUN_ROOT"
echo "summary=$RUN_ROOT/merged/summary.json"
echo "expected_wall_clock=45-150 minutes depending on queue"
