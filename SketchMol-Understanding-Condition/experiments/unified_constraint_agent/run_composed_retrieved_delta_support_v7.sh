#!/usr/bin/env bash
# Train-only support gate for property-observed two-step RetrievedDelta edits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CODE_PROJECT_DIR="$REPO_DIR/SketchMol-Understanding-Condition"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-/home/bdong/.venvs/admet_ai/bin/python}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
V4_ROOT="${SUCC_UCA_V4_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_hierarchical_support_v4}"
V5_ROOT="${SUCC_UCA_V5_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_retrieved_delta_support_v5}"
V6_ROOT="${SUCC_UCA_V6_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_retrieved_delta_planner_v6}"
CEILING_ROOT="${SUCC_UCA_CEILING_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_support_ceiling_v7}"
RUN_ROOT="${SUCC_UCA_COMPOSED_DELTA_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_composed_delta_support_v7}"

TRAIN_ROWS="$V4_ROOT/data/proposer_train_rows.csv"
AUDIT_ROWS="$V4_ROOT/data/support_audit_disjoint_rows.csv"
V5_ANCHORS="$V5_ROOT/candidate_pool/retrieved_delta_candidates.csv"
V5_BASELINE_SUMMARY="$V5_ROOT/support_gate_complete_v2/summary.json"
V5_ORACLE="$V5_ROOT/oracle_complete_v2/generated_properties.csv"
V6_ORACLE="$V6_ROOT/oracle/seed_1709/generated_properties.csv"
CEILING_ORACLE="$CEILING_ROOT/oracle/generated_properties.csv"
BASE_ORACLE="$PROJECT_DIR/outputs/external_oracle_build_v1/generated_properties.csv"

POOL_DIR="$RUN_ROOT/candidate_pool"
ANCHOR_TOP20="$POOL_DIR/frozen_v5_anchor_top20.csv"
ENUMERATED_CANDIDATES="$POOL_DIR/composed_candidates_top256.csv"
ENUMERATION_MANIFEST="$POOL_DIR/composed_manifest.json"
DIAGNOSTIC_CANDIDATES="$POOL_DIR/composed_top96_diagnostic.csv"
DIAGNOSTIC_MANIFEST="$POOL_DIR/diagnostic_manifest.json"
ORACLE_DIR="$RUN_ROOT/oracle"
ORACLE_CSV="$ORACLE_DIR/generated_properties.csv"
OFFICIAL_DIR="$RUN_ROOT/benchmark_with_oracle"
OFFICIAL_DETAIL="$OFFICIAL_DIR/external_multiproperty_detail.csv"
AUDIT_DIR="$RUN_ROOT/audit"

for path in \
  "$PYTHON_BIN" \
  "$ADMET_PYTHON_BIN" \
  "$TRAIN_ROWS" \
  "$AUDIT_ROWS" \
  "$V5_ANCHORS" \
  "$V5_BASELINE_SUMMARY" \
  "$V5_ORACLE" \
  "$V6_ORACLE" \
  "$CEILING_ORACLE" \
  "$BASE_ORACLE"; do
  [[ -e "$path" ]] || { echo "ERROR: missing composed-delta input: $path" >&2; exit 2; }
done

export PYTHONPATH="$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$POOL_DIR" "$ORACLE_DIR" "$OFFICIAL_DIR" "$AUDIT_DIR"

if [[ ! -f "$ENUMERATED_CANDIDATES" || ! -f "$ENUMERATION_MANIFEST" ]]; then
  echo "=== Expand v5 anchors with property-observed two-step deltas ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/build_composed_retrieved_delta_candidates.py" \
    --train-csv "$TRAIN_ROWS" \
    --eval-csv "$AUDIT_ROWS" \
    --anchor-candidates-csv "$V5_ANCHORS" \
    --output-csv "$ANCHOR_TOP20" \
    --enumerated-output-csv "$ENUMERATED_CANDIDATES" \
    --manifest-json "$ENUMERATION_MANIFEST" \
    --candidate-budget 20 \
    --enumerated-limit 256 \
    --max-steps 2 \
    --beam-size 24 \
    --max-compatible-transforms 512 \
    --max-transforms-per-fragment 16 \
    --min-retrieval-similarity 0.15 \
    --min-source-tanimoto 0.4 \
    --min-core-heavy-atoms 5 \
    --max-variable-heavy-atoms 30
fi

if [[ ! -f "$DIAGNOSTIC_CANDIDATES" || ! -f "$DIAGNOSTIC_MANIFEST" ]]; then
  echo "=== Freeze anchor-preserving top-96 before oracle scoring ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/materialize_retrieved_delta_ceiling_pool.py" \
    --enumerated-candidates-csv "$ENUMERATED_CANDIDATES" \
    --source-manifest-json "$ENUMERATION_MANIFEST" \
    --output-csv "$DIAGNOSTIC_CANDIDATES" \
    --manifest-json "$DIAGNOSTIC_MANIFEST" \
    --candidate-limit 96 \
    --paper-candidate-budget 20 \
    --expected-conditions 50
fi

if [[ ! -f "$ORACLE_CSV" ]]; then
  echo "=== Score only the frozen diagnostic prefix ==="
  SUCC_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
  SUCC_ORACLE_INPUT_CSV="$DIAGNOSTIC_CANDIDATES" \
  SUCC_ORACLE_OUTPUT_CSV="$ORACLE_CSV" \
  SUCC_ORACLE_WORK_DIR="$ORACLE_DIR/work" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="$CEILING_ORACLE,$V5_ORACLE,$V6_ORACLE,$BASE_ORACLE" \
  SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES="bbbp,hia,mutagenicity" \
  bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
fi

"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); m=p["missing_counts"]; assert all(int(v) == 0 for v in m.values()), m' \
  "${ORACLE_CSV%.csv}.summary.json"

if [[ ! -f "$OFFICIAL_DETAIL" ]]; then
  echo "=== Evaluate composed-delta support ceiling ==="
  "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
    --prediction-csv "$DIAGNOSTIC_CANDIDATES" \
    --output-dir "$OFFICIAL_DIR" \
    --generated-properties-csv "$ORACLE_CSV" \
    --source-properties-csv "$ORACLE_CSV" \
    --group-column condition_id \
    --min-source-tanimoto 0.4 \
    --report-title "Composed RetrievedDelta support ceiling (diagnostic only)"
fi

echo "=== Gate support before any common-LLM planner training ==="
"$PYTHON_BIN" "$SCRIPT_DIR/audit_retrieved_delta_ceiling.py" \
  --official-detail-csv "$OFFICIAL_DETAIL" \
  --candidate-manifest-json "$DIAGNOSTIC_MANIFEST" \
  --baseline-support-summary "$V5_BASELINE_SUMMARY" \
  --output-dir "$AUDIT_DIR" \
  --target-property-ceiling 0.80 \
  --target-split-property-ceiling 0.70 \
  --target-strict-ceiling 0.60 \
  --target-split-strict-ceiling 0.50 \
  --expected-conditions 50

echo "Composed-delta support gate ready: $AUDIT_DIR/report.md"
