#!/usr/bin/env bash
# Submit the two single-seed transport-adapter arms in parallel, then merge.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
LOG_DIR="${SUCC_LANG_FLOW_LOG_DIR:-$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/language_grounded_graph_latent_flow_v1}"
MAIL_USER="${SUCC_LANG_FLOW_MAIL_USER:-dongbochen1218@gmail.com}"
ACCOUNT="${SUCC_LANG_FLOW_ACCOUNT:-def-hup-ab}"
mkdir -p "$LOG_DIR"

submit_arm() {
  local arm="$1"
  local time_limit="$2"
  local memory="$3"
  local gres="$4"
  sbatch --parsable \
    --job-name="uca-lang-flow-${arm}" \
    --account="$ACCOUNT" \
    --time="$time_limit" \
    --cpus-per-task=4 \
    --mem="$memory" \
    --gres="gpu:${gres}" \
    --mail-user="$MAIL_USER" \
    --mail-type=FAIL \
    --output="$LOG_DIR/uca-lang-flow-${arm}-%j.log" \
    --wrap="SUCC_LANG_FLOW_ARM='$arm' bash '$SCRIPT_DIR/run_language_grounded_graph_latent_flow.sh'"
}

property_job="$(submit_arm property_memory 01:15:00 20G nvidia_h100_80gb_hbm3_1g.10gb:1)"
llm_job="$(submit_arm common_llm_memory 01:30:00 32G nvidia_h100_80gb_hbm3_2g.20gb:1)"
merge_job="$(sbatch --parsable \
  --job-name=uca-lang-flow-merge \
  --account="$ACCOUNT" \
  --time=00:10:00 \
  --cpus-per-task=1 \
  --mem=4G \
  --dependency="afterany:$property_job:$llm_job" \
  --kill-on-invalid-dep=yes \
  --mail-user="$MAIL_USER" \
  --mail-type=END,FAIL \
  --output="$LOG_DIR/uca-lang-flow-merge-%j.log" \
  --wrap="bash '$SCRIPT_DIR/run_merge_language_grounded_graph_latent_flow.sh'")"

echo "property_memory_job=$property_job"
echo "common_llm_memory_job=$llm_job"
echo "merge_job=$merge_job"
echo "result=$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/language_grounded_graph_latent_flow_v1/seed_2001/merged/summary.json"
