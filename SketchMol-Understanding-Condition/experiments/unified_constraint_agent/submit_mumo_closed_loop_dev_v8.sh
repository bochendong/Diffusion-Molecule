#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
LOG_DIR="${SUCC_UCA_LOG_DIR:-$PROJECT_DIR/logs/unified_constraint_agent_mumo_closed_loop_dev_v8/seed_1711}"
MAIL_USER="${SUCC_UCA_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_UCA_ACCOUNT:-def-hup-ab_cpu}"
SHARD_COUNT="${SUCC_UCA_MUMO_DEV_SHARD_COUNT:-16}"
PREPARE_JOB_ID="${SUCC_UCA_REUSE_PREPARE_JOB_ID:-}"
mkdir -p "$LOG_DIR"
parse_id() { printf '%s\n' "$1" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1; }
if [[ -n "$PREPARE_JOB_ID" ]]; then
  [[ "$PREPARE_JOB_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid reused prepare job ID" >&2; exit 2; }
  PREPARE_MANIFEST="$PROJECT_DIR/outputs/unified_constraint_agent_mumo_closed_loop_dev_v8/seed_1711/data/dev_sources.manifest.json"
  [[ -s "$PREPARE_MANIFEST" ]] || { echo "ERROR: reused prepare manifest missing" >&2; exit 2; }
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["generation_target_access"] is False; assert p["target_fields_written"] == 0; assert p["rows"] == 1049' "$PREPARE_MANIFEST"
  prepare_id="$PREPARE_JOB_ID"
else
  prepare="$(sbatch --parsable --account="$ACCOUNT" --job-name=uca-mumo-dev-prepare --time=00:20:00 --cpus-per-task=1 --mem=4G --mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL --output="$LOG_DIR/%x-%j.log" --export=ALL --wrap="bash '$SCRIPT_DIR/run_mumo_closed_loop_dev_v8.sh' prepare")"
  prepare_id="$(parse_id "$prepare")"
fi
array_end="$((SHARD_COUNT - 1))"
generate="$(sbatch --parsable --account="$ACCOUNT" --job-name=uca-mumo-dev-generate --dependency="afterok:$prepare_id" --kill-on-invalid-dep=yes --array="0-${array_end}%${SHARD_COUNT}" --time=02:00:00 --cpus-per-task=4 --mem=12G --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%A_%a.log" --export=ALL,SUCC_UCA_MUMO_DEV_SHARD_COUNT="$SHARD_COUNT" --wrap="bash '$SCRIPT_DIR/run_mumo_closed_loop_dev_v8.sh' generate")"
generate_id="$(parse_id "$generate")"
merge="$(sbatch --parsable --account="$ACCOUNT" --job-name=uca-mumo-dev-merge --dependency="afterok:$generate_id" --kill-on-invalid-dep=yes --time=00:20:00 --cpus-per-task=2 --mem=8G --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" --export=ALL,SUCC_UCA_MUMO_DEV_SHARD_COUNT="$SHARD_COUNT" --wrap="bash '$SCRIPT_DIR/run_mumo_closed_loop_dev_v8.sh' merge")"
merge_id="$(parse_id "$merge")"
oracle="$(sbatch --parsable --account="$ACCOUNT" --job-name=uca-mumo-dev-oracle --dependency="afterok:$merge_id" --kill-on-invalid-dep=yes --time=04:00:00 --cpus-per-task=8 --mem=32G --mail-user="$MAIL_USER" --mail-type=FAIL --output="$LOG_DIR/%x-%j.log" --export=ALL --wrap="bash '$SCRIPT_DIR/run_mumo_closed_loop_dev_v8.sh' oracle")"
oracle_id="$(parse_id "$oracle")"
gate="$(sbatch --parsable --account="$ACCOUNT" --job-name=uca-mumo-dev-gate --dependency="afterok:$oracle_id" --kill-on-invalid-dep=yes --time=00:20:00 --cpus-per-task=2 --mem=8G --mail-user="$MAIL_USER" --mail-type=BEGIN,END,FAIL --output="$LOG_DIR/%x-%j.log" --export=ALL --wrap="bash '$SCRIPT_DIR/run_mumo_closed_loop_dev_v8.sh' gate")"
gate_id="$(parse_id "$gate")"
echo "prepare_job=$prepare_id"
echo "generate_array_job=$generate_id"
echo "merge_job=$merge_id"
echo "oracle_job=$oracle_id"
echo "gate_job=$gate_id"
echo "candidate_budget=20"
echo "accelerator=none"
