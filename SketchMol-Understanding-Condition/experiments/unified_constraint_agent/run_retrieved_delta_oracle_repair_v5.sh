#!/usr/bin/env bash
# Complete the v5 DRD2 oracle without rebuilding candidates or rerunning ADMET-AI.

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
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
V4_ROOT="${SUCC_UCA_V4_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_hierarchical_support_v4}"
RUN_ROOT="${SUCC_UCA_RETRIEVED_DELTA_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_retrieved_delta_support_v5}"
CANDIDATES="$RUN_ROOT/candidate_pool/retrieved_delta_candidates.csv"
CANDIDATE_MANIFEST="$RUN_ROOT/candidate_pool/retrieved_delta_manifest.json"
OLD_ORACLE="$RUN_ROOT/oracle/generated_properties.csv"
BASE_ORACLE="$PROJECT_DIR/outputs/external_oracle_build_v1/generated_properties.csv"
REPAIR_ORACLE_DIR="$RUN_ROOT/oracle_complete_v2"
REPAIR_ORACLE="$REPAIR_ORACLE_DIR/generated_properties.csv"
OFFICIAL_DIR="$RUN_ROOT/benchmark_with_oracle_complete_v2"
GATE_DIR="$RUN_ROOT/support_gate_complete_v2"
PROPOSER_TRAIN_ROWS="$V4_ROOT/data/proposer_train_rows.csv"
AUDIT_ROWS="$V4_ROOT/data/support_audit_disjoint_rows.csv"
OFFICIAL_DETAIL="$OFFICIAL_DIR/external_multiproperty_detail.csv"

for path in "$PYTHON_BIN" "$CANDIDATES" "$CANDIDATE_MANIFEST" "$OLD_ORACLE" "$BASE_ORACLE" "$PROPOSER_TRAIN_ROWS" "$AUDIT_ROWS"; do
  [[ -e "$path" ]] || { echo "ERROR: missing v5 oracle-repair input: $path" >&2; exit 2; }
done

export PYTHONPATH="$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$REPAIR_ORACLE_DIR" "$OFFICIAL_DIR" "$GATE_DIR"

if [[ ! -f "$REPAIR_ORACLE" ]]; then
  echo "=== Fill TDC DRD2 only; reuse completed v5 ADMET/local properties ==="
  SUCC_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_ADMET_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_ORACLE_INPUT_CSV="$CANDIDATES" \
  SUCC_ORACLE_OUTPUT_CSV="$REPAIR_ORACLE" \
  SUCC_ORACLE_WORK_DIR="$REPAIR_ORACLE_DIR/work" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="$OLD_ORACLE,$BASE_ORACLE" \
  SUCC_ORACLE_SKIP_ADMET=1 \
  bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
fi

"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); missing=p["missing_counts"]; assert missing.get("drd2", 1) == 0, missing' \
  "${REPAIR_ORACLE%.csv}.summary.json"

if [[ ! -f "$OFFICIAL_DETAIL" ]]; then
  echo "=== Re-evaluate the unchanged fixed n=20 pool with complete oracle coverage ==="
  "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
    --prediction-csv "$CANDIDATES" \
    --output-dir "$OFFICIAL_DIR" \
    --generated-properties-csv "$REPAIR_ORACLE" \
    --source-properties-csv "$REPAIR_ORACLE" \
    --group-column condition_id \
    --min-source-tanimoto 0.4 \
    --report-title "RetrievedDeltaEdit complete-oracle train-only support n=20"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/audit_hierarchical_action_support.py" \
  --proposer-train-csv "$PROPOSER_TRAIN_ROWS" \
  --audit-rows-csv "$AUDIT_ROWS" \
  --official-detail-csv "$OFFICIAL_DETAIL" \
  --candidate-manifest-json "$CANDIDATE_MANIFEST" \
  --output-dir "$GATE_DIR" \
  --candidate-budget 20 \
  --protocol hierarchical_common_agent_retrieved_delta_support_v5 \
  --proposal-budget 0 \
  --method-label "train-only RetrievedDeltaEdit plus frozen v4 fallback; complete DRD2 oracle" \
  --min-property-any-rate 0.20 \
  --min-strict-any-rate 0.05 \
  --min-full-oracle-condition-rate 1.0

echo "RetrievedDeltaEdit complete-oracle gate ready: $GATE_DIR/report.md"
