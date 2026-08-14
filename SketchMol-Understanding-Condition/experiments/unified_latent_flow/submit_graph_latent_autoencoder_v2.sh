#!/usr/bin/env bash
# Submit one bounded H100-20GB complete-schema graph representation gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SEED="${SUCC_GRAPH_LATENT_SEED:-1725}"
LOG_DIR="${SUCC_GRAPH_LATENT_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/graph_latent_autoencoder_v2}"
MAIL_USER="${SUCC_GRAPH_LATENT_MAIL_USER:-dongbochen1218@gmail.com}"

mkdir -p "$LOG_DIR"

submission="$(sbatch \
  --job-name="uca-graph-schema-s${SEED}" \
  --account=def-hup-ab \
  --time="${SUCC_GRAPH_LATENT_TIME:-00:30:00}" \
  --cpus-per-task="${SUCC_GRAPH_LATENT_CPUS:-4}" \
  --mem="${SUCC_GRAPH_LATENT_MEM:-16G}" \
  --gres="${SUCC_GRAPH_LATENT_GRES:-gpu:nvidia_h100_80gb_hbm3_2g.20gb:1}" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$LOG_DIR/uca-graph-schema-s${SEED}-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_graph_latent_autoencoder_v2.sh'")"

job_id="$(printf '%s\n' "$submission" | awk '{print $NF}')"
echo "$submission"
echo "graph_latent_schema_job=$job_id"
echo "log=$LOG_DIR/uca-graph-schema-s${SEED}-${job_id}.log"
echo "summary=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/graph_latent_autoencoder_v2/seed_${SEED}/summary.json"
