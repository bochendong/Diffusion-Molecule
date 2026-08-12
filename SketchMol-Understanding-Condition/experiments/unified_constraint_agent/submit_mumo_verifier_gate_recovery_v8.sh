#!/usr/bin/env bash
# Reuse completed v8 data/delta/features and rerun only verifier + evidence gate.

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

[[ -s "$RUN_ROOT/data/manifest.json" ]] || { echo "ERROR: missing prepare manifest" >&2; exit 2; }
for index in $(seq 0 $((SHARD_COUNT - 1))); do
  padded="$(printf '%03d' "$index")"
  [[ -s "$RUN_ROOT/delta/manifest_${padded}.json" ]] || { echo "ERROR: missing delta shard $padded" >&2; exit 2; }
  [[ -s "$RUN_ROOT/delta/transforms_${padded}.jsonl" ]] || { echo "ERROR: missing delta transforms $padded" >&2; exit 2; }
  [[ -s "$RUN_ROOT/features/manifest_${padded}.json" ]] || { echo "ERROR: missing feature shard $padded" >&2; exit 2; }
  [[ -s "$RUN_ROOT/features/features_${padded}.npz" ]] || { echo "ERROR: missing feature payload $padded" >&2; exit 2; }
done

mkdir -p "$LOG_DIR"
export SUCC_UCA_SHARED_REPO_DIR="$SHARED_REPO_DIR"
export SUCC_UCA_MUMO_PARALLEL_ROOT="$RUN_ROOT"
export SUCC_UCA_MUMO_SHARDS="$SHARD_COUNT"

parse_job_id() {
  printf '%s\n' "$1" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1
}

echo "Submitting MuMO v8 verifier/gate recovery"
echo "  reuse_prepare=true"
echo "  reuse_delta_shards=32"
echo "  reuse_feature_shards=32"
echo "  candidate_budget=20"
echo "  accelerator=none"

verifier_out="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name=uca-mumo-v8-verifier \
  --array="0-4%5" \
  --time="${SUCC_UCA_VERIFIER_TIME:-00:30:00}" \
  --cpus-per-task=8 \
  --mem=24G \
  --mail-user="$MAIL_USER" \
  --mail-type=FAIL \
  --output="$LOG_DIR/%x-%A_%a.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_mumo_parallel_evidence_v8.sh' verifier")"
verifier_job="$(parse_job_id "$verifier_out")"
[[ -n "$verifier_job" ]] || { echo "ERROR: could not parse verifier array: $verifier_out" >&2; exit 2; }

gate_out="$(sbatch \
  --parsable \
  --account="$ACCOUNT" \
  --job-name=uca-mumo-v8-gate \
  --dependency="afterok:$verifier_job" \
  --kill-on-invalid-dep=yes \
  --time="${SUCC_UCA_GATE_TIME:-00:30:00}" \
  --cpus-per-task=2 \
  --mem=16G \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/%x-%j.log" \
  --export=ALL \
  --wrap="bash '$SCRIPT_DIR/run_mumo_parallel_evidence_v8.sh' finalize")"
gate_job="$(parse_job_id "$gate_out")"
[[ -n "$gate_job" ]] || { echo "ERROR: could not parse gate job: $gate_out" >&2; exit 2; }

echo "verifier_array_job=$verifier_job"
echo "evidence_gate_job=$gate_job"
echo "summary=$RUN_ROOT/merged/summary.json"
