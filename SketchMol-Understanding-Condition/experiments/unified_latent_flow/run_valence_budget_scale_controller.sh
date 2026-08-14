#!/usr/bin/env bash
# Submit a longer B10 scale run only when the matched pilot clears scientific gates.

set -euo pipefail

SUMMARY_PATH="${1:?pilot summary path is required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PYTHON_BIN="${SUCC_VALENCE_HIER_VQ_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
MAIL_USER="${SUCC_VALENCE_HIER_VQ_MAIL_USER:-dongbochen1218@gmail.com}"
SCALE_LOG_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition/logs/valence_budget_hierarchical_vq_graph_flow_v10_scale"
SCALE_OUTPUT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition/outputs/valence_budget_hierarchical_vq_graph_flow_v10_scale/seed_1741"

decision="$($PYTHON_BIN -c 'import json,sys; d=json.load(open(sys.argv[1])); e=d["evaluation"]; by=e.get("by_property_count",{}); ok=(e.get("attempted_per_condition")==20 and e.get("validity",0)>=0.80 and e.get("strict_any20",0)>=0.65 and by.get("3",{}).get("strict_any20",0)>=0.50); print("PASS" if ok else "STOP")' "$SUMMARY_PATH")"
echo "matched_scientific_gate=$decision"
if [[ "$decision" != "PASS" ]]; then
  echo "Scale run not submitted: require validity>=0.80, strict>=0.65, 3p strict>=0.50."
  exit 0
fi

mkdir -p "$SCALE_LOG_DIR"
submission="$(sbatch \
  --job-name="uca-valence-scale-s1741" \
  --account=def-hup-ab \
  --time="${SUCC_VALENCE_HIER_VQ_SCALE_TIME:-02:00:00}" \
  --cpus-per-task=4 \
  --mem=16G \
  --gres="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" \
  --mail-user="$MAIL_USER" \
  --mail-type=BEGIN,END,FAIL \
  --output="$SCALE_LOG_DIR/uca-valence-scale-s1741-%j.log" \
  --export="ALL,SUCC_VALENCE_HIER_VQ_OUTPUT_DIR=$SCALE_OUTPUT_DIR,SUCC_VALENCE_HIER_VQ_TRAIN_LIMIT=10000,SUCC_VALENCE_HIER_VQ_EPOCHS=16" \
  --wrap="bash '$SCRIPT_DIR/run_valence_budget_hierarchical_vq_graph_flow_pilot.sh'")"
echo "$submission"
echo "scale_summary=$SCALE_OUTPUT_DIR/summary.json"
