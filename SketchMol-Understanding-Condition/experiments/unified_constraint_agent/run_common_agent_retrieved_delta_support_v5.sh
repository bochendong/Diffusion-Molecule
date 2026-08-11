#!/usr/bin/env bash
# Train-only support gate for RetrievedDeltaEdit at fixed final oracle n=20.

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
RUN_ROOT="${SUCC_UCA_RETRIEVED_DELTA_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_retrieved_delta_support_v5}"
POOL_DIR="$RUN_ROOT/candidate_pool"
ORACLE_DIR="$RUN_ROOT/oracle"
OFFICIAL_DIR="$RUN_ROOT/benchmark_with_oracle_v1"
GATE_DIR="$RUN_ROOT/support_gate"

PROPOSER_TRAIN_ROWS="${SUCC_UCA_PROPOSER_TRAIN_CSV:-$V4_ROOT/data/proposer_train_rows.csv}"
AUDIT_ROWS="${SUCC_UCA_SUPPORT_AUDIT_CSV:-$V4_ROOT/data/support_audit_disjoint_rows.csv}"
FALLBACK_CANDIDATES="${SUCC_UCA_V4_FALLBACK_CSV:-$V4_ROOT/graph_pool/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv}"
ORACLE_MERGE_CSV="${SUCC_UCA_ORACLE_MERGE_CSV:-$PROJECT_DIR/outputs/external_oracle_build_v1/generated_properties.csv}"
DELTA_CANDIDATES="$POOL_DIR/retrieved_delta_candidates.csv"
DELTA_MANIFEST="$POOL_DIR/retrieved_delta_manifest.json"
ORACLE_CSV="$ORACLE_DIR/generated_properties.csv"
OFFICIAL_DETAIL="$OFFICIAL_DIR/external_multiproperty_detail.csv"
CANDIDATE_BUDGET="${SUCC_UCA_CANDIDATE_BUDGET:-20}"

for path in "$PYTHON_BIN" "$ADMET_PYTHON_BIN" "$PROPOSER_TRAIN_ROWS" "$AUDIT_ROWS" "$FALLBACK_CANDIDATES" "$ORACLE_MERGE_CSV"; do
  [[ -e "$path" ]] || { echo "ERROR: missing RetrievedDeltaEdit input: $path" >&2; exit 2; }
done
if [[ "$CANDIDATE_BUDGET" != "20" ]]; then
  echo "ERROR: the paper-facing RetrievedDeltaEdit protocol fixes final oracle candidate_budget=20" >&2
  exit 2
fi

export PYTHONPATH="$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$POOL_DIR" "$ORACLE_DIR" "$OFFICIAL_DIR" "$GATE_DIR"

echo "=== Validate the frozen zero-overlap v4 train/audit split ==="
"$PYTHON_BIN" "$SCRIPT_DIR/audit_hierarchical_action_support.py" \
  --proposer-train-csv "$PROPOSER_TRAIN_ROWS" \
  --audit-rows-csv "$AUDIT_ROWS" \
  --validate-splits-only

if [[ ! -f "$DELTA_CANDIDATES" || ! -f "$DELTA_MANIFEST" ]]; then
  echo "=== Build train-only matched-pair delta candidates ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/build_retrieved_delta_edit_candidates.py" \
    --train-csv "$PROPOSER_TRAIN_ROWS" \
    --eval-csv "$AUDIT_ROWS" \
    --fallback-candidates-csv "$FALLBACK_CANDIDATES" \
    --output-csv "$DELTA_CANDIDATES" \
    --manifest-json "$DELTA_MANIFEST" \
    --candidate-budget "$CANDIDATE_BUDGET" \
    --min-retrieval-similarity "${SUCC_UCA_MIN_DELTA_RETRIEVAL_SIMILARITY:-0.15}" \
    --max-transforms-per-query "${SUCC_UCA_MAX_DELTA_TRANSFORMS_PER_QUERY:-96}" \
    --min-core-heavy-atoms "${SUCC_UCA_MIN_DELTA_CORE_HEAVY_ATOMS:-5}" \
    --max-variable-heavy-atoms "${SUCC_UCA_MAX_DELTA_VARIABLE_HEAVY_ATOMS:-30}" \
    --min-source-tanimoto 0.4
else
  echo "=== Reuse completed RetrievedDeltaEdit candidate pool ==="
fi

if [[ ! -f "$ORACLE_CSV" ]]; then
  echo "=== Score exactly 20 final candidates per condition with the official oracle ==="
  SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
  SUCC_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_ORACLE_INPUT_CSV="$DELTA_CANDIDATES" \
  SUCC_ORACLE_OUTPUT_CSV="$ORACLE_CSV" \
  SUCC_ORACLE_WORK_DIR="$ORACLE_DIR/work" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="$ORACLE_MERGE_CSV" \
  SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES="bbbp,hia,mutagenicity" \
  bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
else
  echo "=== Reuse completed RetrievedDeltaEdit oracle ==="
fi

if [[ ! -f "$OFFICIAL_DETAIL" ]]; then
  echo "=== Materialize train-only official support metrics ==="
  "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
    --prediction-csv "$DELTA_CANDIDATES" \
    --output-dir "$OFFICIAL_DIR" \
    --generated-properties-csv "$ORACLE_CSV" \
    --source-properties-csv "$ORACLE_CSV" \
    --group-column condition_id \
    --min-source-tanimoto 0.4 \
    --report-title "RetrievedDeltaEdit train-only matched-pair support n=20"
fi

echo "=== Decide whether source-preserving delta support is sufficient ==="
"$PYTHON_BIN" "$SCRIPT_DIR/audit_hierarchical_action_support.py" \
  --proposer-train-csv "$PROPOSER_TRAIN_ROWS" \
  --audit-rows-csv "$AUDIT_ROWS" \
  --official-detail-csv "$OFFICIAL_DETAIL" \
  --candidate-manifest-json "$DELTA_MANIFEST" \
  --output-dir "$GATE_DIR" \
  --candidate-budget "$CANDIDATE_BUDGET" \
  --protocol hierarchical_common_agent_retrieved_delta_support_v5 \
  --proposal-budget 0 \
  --method-label "train-only RetrievedDeltaEdit plus frozen v4 fallback" \
  --min-property-any-rate "${SUCC_UCA_MIN_SUPPORT_PROPERTY_ANY_RATE:-0.20}" \
  --min-strict-any-rate "${SUCC_UCA_MIN_SUPPORT_STRICT_ANY_RATE:-0.05}"

echo "RetrievedDeltaEdit support gate ready: $GATE_DIR/report.md"
