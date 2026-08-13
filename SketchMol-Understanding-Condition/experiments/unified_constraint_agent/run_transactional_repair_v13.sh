#!/usr/bin/env bash
# Reuse the frozen v12 controller and compare transactional repair on MuMO dev.

set -euo pipefail

STAGE="${1:?usage: run_transactional_repair_v13.sh trajectory|merge|oracle|gate}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CODE_PROJECT_DIR="$REPO_DIR/SketchMol-Understanding-Condition"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-/home/bdong/.venvs/admet_ai/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
EVIDENCE_ROOT="$PROJECT_DIR/outputs/unified_constraint_agent_mumo_parallel_evidence_v8/seed_1711"
V8_ROOT="$PROJECT_DIR/outputs/unified_constraint_agent_mumo_closed_loop_dev_v8/seed_1711"
V9_ROOT="$PROJECT_DIR/outputs/unified_constraint_agent_mumo_residual_planner_v9/seed_1712"
V12_ROOT="${SUCC_UCA_V12_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_direct_repair_v12/seed_1715}"
RUN_ROOT="${SUCC_UCA_TRANSACTIONAL_REPAIR_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_transactional_repair_v13/seed_1715}"
DEV_SOURCES="$V8_ROOT/data/dev_sources.jsonl"
DEV_MANIFEST="$V8_ROOT/data/dev_sources.manifest.json"
PLANS="$V12_ROOT/controller/dev_plans.jsonl"
PLAN_MANIFEST="$V12_ROOT/controller/dev_plans.manifest.json"
SHARD_COUNT="${SUCC_UCA_TRANSACTIONAL_REPAIR_SHARD_COUNT:-16}"
TRAJECTORY_DIR="$RUN_ROOT/trajectories"
CANDIDATES="$TRAJECTORY_DIR/exact_n20.csv"
CANDIDATE_MANIFEST="$TRAJECTORY_DIR/manifest.json"
ORACLE="$RUN_ROOT/oracle/generated_properties.csv"
EVAL_DIR="$RUN_ROOT/evaluation"
GATE_DIR="$RUN_ROOT/gate"

export PYTHONPATH="$DEP_OVERLAY:$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TRAJECTORY_DIR/shards" "$RUN_ROOT/oracle" "$EVAL_DIR" "$GATE_DIR"

case "$STAGE" in
  trajectory)
    SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?trajectory requires a Slurm array task}"
    SHARD_TAG="$(printf '%03d' "$SHARD_INDEX")"
    "$PYTHON_BIN" "$SCRIPT_DIR/generate_direct_repair_trajectories.py" \
      --evidence-root "$EVIDENCE_ROOT" \
      --dev-sources-jsonl "$DEV_SOURCES" \
      --plans-jsonl "$PLANS" --plans-manifest "$PLAN_MANIFEST" \
      --output-csv "$TRAJECTORY_DIR/shards/trajectories_${SHARD_TAG}.csv" \
      --manifest-json "$TRAJECTORY_DIR/shards/manifest_${SHARD_TAG}.json" \
      --attempts-per-condition 20 --max-steps 3 --max-transaction-proposals 3 \
      --max-symbolic-actions 256 --action-retry-limit 12 --temperature 0.75 \
      --min-retrieval-similarity 0.15 --min-source-tanimoto 0.4 \
      --min-focus-margin-improvement 0.005 \
      --min-total-violation-improvement 0.005 --max-margin-regression 0.05 \
      --transactional --method common_llm_transactional_constraint_repair_v13 \
      --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" --seed 1715
    ;;
  merge)
    EXPECTED_CONDITIONS="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "$DEV_MANIFEST")"
    "$PYTHON_BIN" "$SCRIPT_DIR/merge_direct_repair_trajectories.py" \
      --shard-dir "$TRAJECTORY_DIR/shards" --output-csv "$CANDIDATES" \
      --manifest-json "$CANDIDATE_MANIFEST" --shard-count "$SHARD_COUNT" \
      --expected-conditions "$EXPECTED_CONDITIONS"
    ;;
  oracle)
    [[ -s "$CANDIDATES" && -s "$CANDIDATE_MANIFEST" ]] || {
      echo "ERROR: exact-20 transactional trajectory output missing" >&2; exit 2;
    }
    SUCC_PYTHON_BIN="$PYTHON_BIN" SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
    SUCC_ORACLE_INPUT_CSV="$CANDIDATES" SUCC_ORACLE_OUTPUT_CSV="$ORACLE" \
    SUCC_ORACLE_WORK_DIR="$RUN_ROOT/oracle/work" \
    SUCC_ORACLE_MERGE_PROPERTIES_CSV="$V12_ROOT/oracle/generated_properties.csv" \
    SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES=bbbp,hia,mutagenicity \
    bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
    "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
      --prediction-csv "$CANDIDATES" --output-dir "$EVAL_DIR" \
      --generated-properties-csv "$ORACLE" --source-properties-csv "$ORACLE" \
      --group-column condition_id --min-source-tanimoto 0.4 \
      --report-title "Common-LLM transactional constraint repair v13 MuMO dev exact n=20"
    ;;
  gate)
    "$PYTHON_BIN" "$SCRIPT_DIR/finalize_direct_repair_gate.py" \
      --trajectory-manifest "$CANDIDATE_MANIFEST" --plan-manifest "$PLAN_MANIFEST" \
      --controller-data-manifest "$V12_ROOT/controller/data/manifest.json" \
      --training-summary "$V12_ROOT/controller/model/training_summary.json" \
      --summary-csv "$EVAL_DIR/external_multiproperty_summary.csv" \
      --oracle-summary "${ORACLE%.csv}.summary.json" \
      --baseline-gate "$V8_ROOT/gate/summary.json" \
      --baseline-format-summary "$V9_ROOT/anti_forgetting/baseline/summary.json" \
      --candidate-format-summary "$V12_ROOT/anti_forgetting/candidate/summary.json" \
      --output-dir "$GATE_DIR" --require-transactional
    ;;
  *) echo "ERROR: unknown stage $STAGE" >&2; exit 2 ;;
esac
