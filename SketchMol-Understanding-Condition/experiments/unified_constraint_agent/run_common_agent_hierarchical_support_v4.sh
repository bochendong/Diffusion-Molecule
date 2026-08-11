#!/usr/bin/env bash
# Train-only support gate for a hierarchical common agent: raw-1 proposal -> GraphEditDSL -> oracle n=20.

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
RUN_ROOT="${SUCC_UCA_HIERARCHICAL_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_hierarchical_support_v4}"
DATA_DIR="$RUN_ROOT/data"
PROPOSER_DIR="$RUN_ROOT/proposer"
GRAPH_DIR="$RUN_ROOT/graph_pool"
GRAPH_BENCHMARK_DIR="$GRAPH_DIR/benchmark_graph_edit_agent"
ORACLE_DIR="$GRAPH_DIR/oracle"
OFFICIAL_DIR="$GRAPH_DIR/benchmark_with_oracle_v1"
GATE_DIR="$RUN_ROOT/support_gate"

MUMO_TRAIN_JSON="${SUCC_UCA_MUMO_TRAIN_JSON:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/train.json}"
PROPOSER_BASE_CHECKPOINT="${SUCC_UCA_PROPOSER_BASE_CHECKPOINT:-$PROJECT_DIR/outputs/direct_smiles_moledit_table1_group_rl_v1/direct_smiles_model_source_edit_sft/direct_smiles_generator.pt}"
ORACLE_MERGE_CSV="${SUCC_UCA_ORACLE_MERGE_CSV:-$PROJECT_DIR/outputs/external_oracle_build_v1/generated_properties.csv}"

PROPOSER_TRAIN_ROWS="$DATA_DIR/proposer_train_rows.csv"
AUDIT_CANDIDATE_ROWS="$DATA_DIR/support_audit_candidate_rows.csv"
AUDIT_ROWS="$DATA_DIR/support_audit_disjoint_rows.csv"
DIRECT_PROPOSALS="$PROPOSER_DIR/raw1_proposals.csv"
PROPOSER_CHECKPOINT="$PROPOSER_DIR/direct_smiles_generator.pt"
GRAPH_CANDIDATES="$GRAPH_BENCHMARK_DIR/graph_edit_agent_candidate_predictions.csv"
GRAPH_SELECTED="$GRAPH_BENCHMARK_DIR/graph_edit_agent_predictions.csv"
GRAPH_PLANS="$GRAPH_DIR/graph_edit_plans.jsonl"
ORACLE_CSV="$ORACLE_DIR/generated_properties.csv"
OFFICIAL_DETAIL="$OFFICIAL_DIR/external_multiproperty_detail.csv"

PROPOSER_ROWS_PER_TASK="${SUCC_UCA_PROPOSER_ROWS_PER_TASK:-100}"
AUDIT_ROWS_PER_TASK="${SUCC_UCA_SUPPORT_ROWS_PER_TASK:-5}"
AUDIT_CANDIDATE_ROWS_PER_TASK="${SUCC_UCA_SUPPORT_CANDIDATE_ROWS_PER_TASK:-100}"
PROPOSER_SEED="${SUCC_UCA_PROPOSER_SEED:-1711}"
AUDIT_SEED="${SUCC_UCA_SUPPORT_SEED:-1712}"
CANDIDATE_BUDGET="${SUCC_UCA_CANDIDATE_BUDGET:-20}"

for path in "$PYTHON_BIN" "$ADMET_PYTHON_BIN" "$MUMO_TRAIN_JSON" "$PROPOSER_BASE_CHECKPOINT" "$ORACLE_MERGE_CSV"; do
  [[ -e "$path" ]] || { echo "ERROR: missing hierarchical support input: $path" >&2; exit 2; }
done
if [[ "$CANDIDATE_BUDGET" != "20" ]]; then
  echo "ERROR: the paper-facing hierarchical support protocol fixes final oracle candidate_budget=20" >&2
  exit 2
fi

export PYTHONPATH="$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$DATA_DIR" "$PROPOSER_DIR" "$GRAPH_BENCHMARK_DIR" "$ORACLE_DIR" "$OFFICIAL_DIR" "$GATE_DIR"

export_rows() {
  local output_csv="$1"
  local summary_json="$2"
  local task_spec_json="$3"
  local rows_per_task="$4"
  local seed="$5"
  if [[ ! -f "$output_csv" ]]; then
    "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/export_external_multiproperty_benchmark_rows.py" \
      --source-file "$MUMO_TRAIN_JSON" \
      --output-csv "$output_csv" \
      --summary-json "$summary_json" \
      --task-spec-json "$task_spec_json" \
      --suite mumo \
      --task-split all \
      --input-split train \
      --max-rows-per-task "$rows_per_task" \
      --seed "$seed"
  fi
}

echo "=== Export disjoint proposer-train and support-audit rows ==="
export_rows \
  "$PROPOSER_TRAIN_ROWS" \
  "$DATA_DIR/proposer_train_rows.summary.json" \
  "$DATA_DIR/proposer_train_rows.task_specs.json" \
  "$PROPOSER_ROWS_PER_TASK" \
  "$PROPOSER_SEED"
export_rows \
  "$AUDIT_CANDIDATE_ROWS" \
  "$DATA_DIR/support_audit_candidate_rows.summary.json" \
  "$DATA_DIR/support_audit_candidate_rows.task_specs.json" \
  "$AUDIT_CANDIDATE_ROWS_PER_TASK" \
  "$AUDIT_SEED"
if [[ ! -f "$AUDIT_ROWS" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/select_disjoint_support_rows.py" \
    --proposer-train-csv "$PROPOSER_TRAIN_ROWS" \
    --audit-candidate-csv "$AUDIT_CANDIDATE_ROWS" \
    --output-csv "$AUDIT_ROWS" \
    --manifest-json "$DATA_DIR/support_audit_disjoint_rows.manifest.json" \
    --rows-per-task "$AUDIT_ROWS_PER_TASK"
fi
"$PYTHON_BIN" "$SCRIPT_DIR/audit_hierarchical_action_support.py" \
  --proposer-train-csv "$PROPOSER_TRAIN_ROWS" \
  --audit-rows-csv "$AUDIT_ROWS" \
  --validate-splits-only

if [[ ! -f "$PROPOSER_CHECKPOINT" || ! -f "$DIRECT_PROPOSALS" ]]; then
  echo "=== Fit the train-only source-conditioned proposal tool and emit raw-1 audit proposals ==="
  "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/train_direct_smiles_generator.py" \
    --train-csv "$PROPOSER_TRAIN_ROWS" \
    --eval-csv "$AUDIT_ROWS" \
    --resume-checkpoint "$PROPOSER_BASE_CHECKPOINT" \
    --condition-mixing-mode append_source_property_program \
    --output-dir "$PROPOSER_DIR" \
    --prediction-csv "$DIRECT_PROPOSALS" \
    --epochs 1 \
    --batch-size "${SUCC_UCA_PROPOSER_BATCH_SIZE:-32}" \
    --eval-batch-size "${SUCC_UCA_PROPOSER_EVAL_BATCH_SIZE:-32}" \
    --lr "${SUCC_UCA_PROPOSER_LR:-1e-5}" \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --max-smiles-length 160 \
    --max-new-tokens 100 \
    --temperature 0.70 \
    --top-k 24 \
    --top-p 0.90 \
    --num-samples 1 \
    --parallel-samples 1 \
    --max-parallel-sequences 128 \
    --repetition-penalty 1.15 \
    --no-repeat-ngram-size 6 \
    --min-new-tokens 6 \
    --disable-property-rerank \
    --reset-training-state \
    --seed "$PROPOSER_SEED" \
    --device cuda
else
  echo "=== Reuse completed raw-1 proposal tool artifacts ==="
fi

if [[ ! -f "$GRAPH_CANDIDATES" || ! -f "$GRAPH_PLANS" ]]; then
  echo "=== Build two-step proposal-conditioned GraphEditDSL pool with final n=20 ==="
  "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/build_external_graph_edit_agent_predictions.py" \
    --rows-csv "$AUDIT_ROWS" \
    --direct-prediction-csv "$DIRECT_PROPOSALS" \
    --prediction-csv "$GRAPH_SELECTED" \
    --candidate-output-csv "$GRAPH_CANDIDATES" \
    --plan-jsonl "$GRAPH_PLANS" \
    --planner-mode heuristic_graph_dsl \
    --selection-mode similarity_first \
    --min-source-tanimoto 0.4 \
    --planner-steps 2 \
    --beam-size "${SUCC_UCA_SUPPORT_BEAM_SIZE:-64}" \
    --site-limit "${SUCC_UCA_SUPPORT_SITE_LIMIT:-32}" \
    --max-plans-per-property "${SUCC_UCA_SUPPORT_MAX_PLANS_PER_PROPERTY:-160}" \
    --max-candidates-per-parent "${SUCC_UCA_SUPPORT_MAX_CANDIDATES_PER_PARENT:-160}" \
    --max-candidates-per-row "${SUCC_UCA_SUPPORT_MAX_CANDIDATES_PER_ROW:-2048}" \
    --similarity-first-min-local-success-fraction 0.9 \
    --property-weight 80 \
    --distance-weight 8 \
    --similarity-weight 80 \
    --similarity-bonus 180 \
    --copy-penalty 10 \
    --top-k-candidates "$CANDIDATE_BUDGET" \
    --checkpoint \
    --checkpoint-dir "$GRAPH_BENCHMARK_DIR/checkpoints" \
    --checkpoint-every 5 \
    --resume \
    --seed "$AUDIT_SEED"
else
  echo "=== Reuse completed two-step graph pool ==="
fi

if [[ ! -f "$ORACLE_CSV" ]]; then
  echo "=== Score exactly n=20 final molecules with complete ADMET-AI + TDC oracle ==="
  SUCC_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
  SUCC_ORACLE_INPUT_CSV="$GRAPH_CANDIDATES" \
  SUCC_ORACLE_OUTPUT_CSV="$ORACLE_CSV" \
  SUCC_ORACLE_WORK_DIR="$ORACLE_DIR/work" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="$ORACLE_MERGE_CSV" \
  SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES="bbbp,hia,mutagenicity" \
  bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
else
  echo "=== Reuse completed official oracle ==="
fi

if [[ ! -f "$OFFICIAL_DETAIL" ]]; then
  echo "=== Materialize train-only official support metrics ==="
  "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
    --prediction-csv "$GRAPH_CANDIDATES" \
    --output-dir "$OFFICIAL_DIR" \
    --generated-properties-csv "$ORACLE_CSV" \
    --source-properties-csv "$ORACLE_CSV" \
    --group-column condition_id \
    --min-source-tanimoto 0.4 \
    --report-title "Hierarchical common-agent train-only raw-1 proposal plus GraphEditDSL n=20"
fi

echo "=== Decide whether proposal-plus-edit support is sufficient for planner training ==="
"$PYTHON_BIN" "$SCRIPT_DIR/audit_hierarchical_action_support.py" \
  --proposer-train-csv "$PROPOSER_TRAIN_ROWS" \
  --audit-rows-csv "$AUDIT_ROWS" \
  --official-detail-csv "$OFFICIAL_DETAIL" \
  --output-dir "$GATE_DIR" \
  --candidate-budget "$CANDIDATE_BUDGET" \
  --min-property-any-rate "${SUCC_UCA_MIN_SUPPORT_PROPERTY_ANY_RATE:-0.20}" \
  --min-strict-any-rate "${SUCC_UCA_MIN_SUPPORT_STRICT_ANY_RATE:-0.05}"

echo "Hierarchical common-agent support gate ready: $GATE_DIR/report.md"
