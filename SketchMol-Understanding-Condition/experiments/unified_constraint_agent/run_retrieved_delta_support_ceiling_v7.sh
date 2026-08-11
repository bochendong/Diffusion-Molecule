#!/usr/bin/env bash
# CPU-only support ceiling audit for the v7 method decision.

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
V5_ROOT="${SUCC_UCA_V5_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_retrieved_delta_support_v5}"
V6_ROOT="${SUCC_UCA_V6_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_retrieved_delta_planner_v6}"
RUN_ROOT="${SUCC_UCA_CEILING_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_support_ceiling_v7}"

ENUMERATED_CANDIDATES="$V6_ROOT/candidate_pool/enumerated_candidates.csv"
ENUMERATION_MANIFEST="$V6_ROOT/candidate_pool/enumeration_manifest.json"
V5_BASELINE_SUMMARY="$V5_ROOT/support_gate_complete_v2/summary.json"
V5_ORACLE="$V5_ROOT/oracle_complete_v2/generated_properties.csv"
V6_ORACLE="$V6_ROOT/oracle/seed_1709/generated_properties.csv"
BASE_ORACLE="$PROJECT_DIR/outputs/external_oracle_build_v1/generated_properties.csv"
POOL_DIR="$RUN_ROOT/candidate_pool"
DIAGNOSTIC_CANDIDATES="$POOL_DIR/heuristic_top96_diagnostic.csv"
DIAGNOSTIC_MANIFEST="$POOL_DIR/manifest.json"
ORACLE_DIR="$RUN_ROOT/oracle"
ORACLE_CSV="$ORACLE_DIR/generated_properties.csv"
OFFICIAL_DIR="$RUN_ROOT/benchmark_with_oracle"
OFFICIAL_DETAIL="$OFFICIAL_DIR/external_multiproperty_detail.csv"
AUDIT_DIR="$RUN_ROOT/audit"

for path in \
  "$PYTHON_BIN" \
  "$ADMET_PYTHON_BIN" \
  "$ENUMERATED_CANDIDATES" \
  "$ENUMERATION_MANIFEST" \
  "$V5_BASELINE_SUMMARY" \
  "$V5_ORACLE" \
  "$V6_ORACLE" \
  "$BASE_ORACLE"; do
  [[ -e "$path" ]] || { echo "ERROR: missing support-ceiling input: $path" >&2; exit 2; }
done

export PYTHONPATH="$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$POOL_DIR" "$ORACLE_DIR" "$OFFICIAL_DIR" "$AUDIT_DIR"

if [[ ! -f "$DIAGNOSTIC_CANDIDATES" || ! -f "$DIAGNOSTIC_MANIFEST" ]]; then
  echo "=== Freeze the oracle-blind top-96 diagnostic prefix ==="
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
  echo "=== Score the diagnostic prefix after selection (CPU only) ==="
  SUCC_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
  SUCC_ORACLE_INPUT_CSV="$DIAGNOSTIC_CANDIDATES" \
  SUCC_ORACLE_OUTPUT_CSV="$ORACLE_CSV" \
  SUCC_ORACLE_WORK_DIR="$ORACLE_DIR/work" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="$V5_ORACLE,$V6_ORACLE,$BASE_ORACLE" \
  SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES="bbbp,hia,mutagenicity" \
  bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
fi

"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); m=p["missing_counts"]; assert all(int(v) == 0 for v in m.values()), m' \
  "${ORACLE_CSV%.csv}.summary.json"

if [[ ! -f "$OFFICIAL_DETAIL" ]]; then
  echo "=== Evaluate variable-k diagnostic support ==="
  "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
    --prediction-csv "$DIAGNOSTIC_CANDIDATES" \
    --output-dir "$OFFICIAL_DIR" \
    --generated-properties-csv "$ORACLE_CSV" \
    --source-properties-csv "$ORACLE_CSV" \
    --group-column condition_id \
    --min-source-tanimoto 0.4 \
    --report-title "RetrievedDelta oracle-blind support ceiling (diagnostic only)"
fi

echo "=== Decide whether the current support can reach 70% ==="
"$PYTHON_BIN" "$SCRIPT_DIR/audit_retrieved_delta_ceiling.py" \
  --official-detail-csv "$OFFICIAL_DETAIL" \
  --candidate-manifest-json "$DIAGNOSTIC_MANIFEST" \
  --baseline-support-summary "$V5_BASELINE_SUMMARY" \
  --output-dir "$AUDIT_DIR" \
  --target-strict-ceiling 0.70 \
  --expected-conditions 50

echo "Support-ceiling audit ready: $AUDIT_DIR/report.md"
