#!/usr/bin/env bash
# Fixed-n train-dev signal after the v8 train-only evidence gate.

set -euo pipefail

STAGE="${1:?usage: run_mumo_closed_loop_dev_v8.sh prepare|generate|merge|oracle|gate}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CODE_PROJECT_DIR="$REPO_DIR/SketchMol-Understanding-Condition"
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
EVIDENCE_ROOT="${SUCC_UCA_MUMO_PARALLEL_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_mumo_parallel_evidence_v8/seed_1711}"
RUN_ROOT="${SUCC_UCA_MUMO_CLOSED_LOOP_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_mumo_closed_loop_dev_v8/seed_1711}"
DATA_DIR="$EVIDENCE_ROOT/data"
DEV_SOURCES="$RUN_ROOT/data/dev_sources.jsonl"
DEV_SOURCE_MANIFEST="$RUN_ROOT/data/dev_sources.manifest.json"
SHARD_COUNT="${SUCC_UCA_MUMO_DEV_SHARD_COUNT:-16}"
CANDIDATES="$RUN_ROOT/candidates/fixed_n20.csv"
ENUMERATED="$RUN_ROOT/candidates/internal_top48.csv"
MANIFEST="$RUN_ROOT/candidates/manifest.json"
ORACLE="$RUN_ROOT/oracle/generated_properties.csv"
EVAL_DIR="$RUN_ROOT/evaluation"
export PYTHONPATH="$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT/data" "$RUN_ROOT/candidates/shards" "$RUN_ROOT/oracle" "$EVAL_DIR" "$RUN_ROOT/gate"

case "$STAGE" in
  prepare)
    "$PYTHON_BIN" "$SCRIPT_DIR/materialize_mumo_dev_sources.py" \
      --data-dir "$DATA_DIR" \
      --output-jsonl "$DEV_SOURCES" \
      --manifest-json "$DEV_SOURCE_MANIFEST"
    ;;
  generate)
    SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?generate requires a Slurm array task}"
    SHARD_TAG="$(printf '%03d' "$SHARD_INDEX")"
    "$PYTHON_BIN" "$SCRIPT_DIR/build_mumo_closed_loop_dev.py" \
      --run-root "$EVIDENCE_ROOT" \
      --dev-sources-jsonl "$DEV_SOURCES" \
      --output-csv "$RUN_ROOT/candidates/shards/candidates_${SHARD_TAG}.csv" \
      --enumerated-output-csv "$RUN_ROOT/candidates/shards/enumerated_${SHARD_TAG}.csv" \
      --manifest-json "$RUN_ROOT/candidates/shards/manifest_${SHARD_TAG}.json" \
      --candidate-budget 20 \
      --planner-candidate-limit 48 \
      --shard-index "$SHARD_INDEX" \
      --shard-count "$SHARD_COUNT"
    ;;
  merge)
    EXPECTED_CONDITIONS="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "$DEV_SOURCE_MANIFEST")"
    "$PYTHON_BIN" "$SCRIPT_DIR/merge_mumo_closed_loop_dev.py" \
      --shard-dir "$RUN_ROOT/candidates/shards" \
      --output-csv "$CANDIDATES" \
      --enumerated-output-csv "$ENUMERATED" \
      --manifest-json "$MANIFEST" \
      --shard-count "$SHARD_COUNT" \
      --expected-conditions "$EXPECTED_CONDITIONS"
    ;;
  oracle)
    [[ -s "$CANDIDATES" && -s "$MANIFEST" ]] || { echo "ERROR: fixed n=20 pool missing" >&2; exit 2; }
    SUCC_PYTHON_BIN="$PYTHON_BIN" \
    SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
    SUCC_ORACLE_INPUT_CSV="$CANDIDATES" \
    SUCC_ORACLE_OUTPUT_CSV="$ORACLE" \
    SUCC_ORACLE_WORK_DIR="$RUN_ROOT/oracle/work" \
    SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES="bbbp,hia,mutagenicity" \
    bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
    "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
      --prediction-csv "$CANDIDATES" \
      --output-dir "$EVAL_DIR" \
      --generated-properties-csv "$ORACLE" \
      --source-properties-csv "$ORACLE" \
      --group-column condition_id \
      --min-source-tanimoto 0.4 \
      --report-title "MuMO fit-only pair-verifier closed-loop dev n=20"
    ;;
  gate)
    "$PYTHON_BIN" "$SCRIPT_DIR/finalize_mumo_closed_loop_dev.py" \
      --candidate-manifest "$MANIFEST" \
      --summary-csv "$EVAL_DIR/external_multiproperty_summary.csv" \
      --oracle-summary "${ORACLE%.csv}.summary.json" \
      --output-json "$RUN_ROOT/gate/summary.json" \
      --min-sr 0.65 \
      --min-ood-sr 0.60 \
      --min-validity 0.95
    ;;
  *) echo "ERROR: unknown stage $STAGE" >&2; exit 2 ;;
esac
