#!/usr/bin/env bash
# Submit one bounded CPU-only B36 representation evidence job.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_GRAPH_PATCH_SEED:-1981}"
LOG_DIR="${SUCC_GRAPH_PATCH_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/source_anchored_graph_patch_evidence_v36}"
MAIL_USER="${SUCC_GRAPH_PATCH_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --job-name="uca-graph-patch-v36" \
  --account=def-hup-ab_cpu \
  --time="${SUCC_GRAPH_PATCH_TIME:-00:45:00}" \
  --cpus-per-task="${SUCC_GRAPH_PATCH_CPUS:-4}" \
  --mem="${SUCC_GRAPH_PATCH_MEM:-20G}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-graph-patch-v36-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_source_anchored_graph_patch_evidence.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "graph_patch_evidence_job=$job_id"
echo "log=$LOG_DIR/uca-graph-patch-v36-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/source_anchored_graph_patch_evidence_v36/seed_${SEED}/summary.json"
