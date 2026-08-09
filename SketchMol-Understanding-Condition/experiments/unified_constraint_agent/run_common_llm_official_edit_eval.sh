#!/usr/bin/env bash
# Run frozen common-LLM GraphEditDSL evaluation on one official edit suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

SUITE="${1:-${SUCC_UCA_OFFICIAL_SUITE:-}}"
if [[ "$SUITE" != "table1" && "$SUITE" != "mumo" ]]; then
  echo "ERROR: suite must be table1 or mumo (got '$SUITE')" >&2
  exit 2
fi

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
ADMET_PYTHON_BIN="${SUCC_ADMET_PYTHON_BIN:-/home/bdong/.venvs/admet_ai/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
CODE_PROJECT_DIR="$REPO_DIR/SketchMol-Understanding-Condition"
PREF_ROOT="${SUCC_UCA_VERIFIER_PREFERENCE_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_verifier_preference_v2}"
ADAPTER_DIR="${SUCC_UCA_OFFICIAL_ADAPTER:-$PREF_ROOT/model/seed_1705/adapter}"
OUTPUT_ROOT="${SUCC_UCA_OFFICIAL_OUTPUT_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_official_edit_v1}"
OUTPUT_DIR="$OUTPUT_ROOT/$SUITE"
BASE_MODEL="${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TABLE1_INPUT="${SUCC_UCA_TABLE1_INPUT:-$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_test_rows.csv}"
MUMO_INPUT="${SUCC_UCA_MUMO_INPUT:-$PROJECT_DIR/outputs/external_mumo_official_graph_edit_heuristic_2step_v1/external_multiproperty_rows.csv}"
INPUT_CSV="$TABLE1_INPUT"
[[ "$SUITE" == "mumo" ]] && INPUT_CSV="$MUMO_INPUT"

export PYTHONPATH="$DEP_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false

for path in "$INPUT_CSV" "$ADAPTER_DIR/adapter_model.safetensors"; do
  [[ -f "$path" ]] || { echo "ERROR: missing official-eval input: $path" >&2; exit 2; }
done
mkdir -p "$OUTPUT_DIR"

echo "Common-LLM official edit evaluation"
echo "  suite=$SUITE"
echo "  input_csv=$INPUT_CSV"
echo "  adapter_dir=$ADAPTER_DIR"
echo "  output_dir=$OUTPUT_DIR"
echo "  candidate_budget=20"
echo "  verifier_k=5"

eval_args=(
  "$SCRIPT_DIR/evaluate_common_llm_official_actions.py"
  --input-csv "$INPUT_CSV"
  --output-dir "$OUTPUT_DIR/ranking"
  --suite "$SUITE"
  --base-model "$BASE_MODEL"
  --adapter-dir "$ADAPTER_DIR"
  --variant verifier_preference_seed_1705
  --candidate-budget 20
  --verifier-k 5
  --enumeration-attempt-budget "${SUCC_UCA_ENUMERATION_ATTEMPT_BUDGET:-64}"
  --max-enumeration-attempt-budget "${SUCC_UCA_MAX_ENUMERATION_ATTEMPT_BUDGET:-512}"
  --site-limit "${SUCC_UCA_SITE_LIMIT:-32}"
  --score-batch-size "${SUCC_UCA_SCORE_BATCH_SIZE:-4}"
  --max-length "${SUCC_UCA_EVAL_MAX_LENGTH:-1024}"
)
if [[ -n "${SUCC_UCA_OFFICIAL_MAX_ROWS:-}" ]]; then
  eval_args+=(--max-rows "$SUCC_UCA_OFFICIAL_MAX_ROWS")
fi
"$PYTHON_BIN" "${eval_args[@]}"

if [[ "$SUITE" == "table1" ]]; then
  echo "=== Official MolEdit Table1 scoring at n=1,5,20 ==="
  SUCC_UNIFIED_BENCHMARK_RUN_SAMPLE=0 \
  SUCC_UNIFIED_BENCHMARK_CANDIDATE_CSV="$OUTPUT_DIR/ranking/candidates.csv" \
  SUCC_UNIFIED_BENCHMARK_OUTPUT_DIR="$OUTPUT_DIR/official" \
  SUCC_UNIFIED_BENCHMARK_TASKS=moledit_table1 \
  SUCC_UNIFIED_METHOD_NAME=common_llm_graph_edit_seed1705 \
  SUCC_UNIFIED_CANDIDATE_BUDGETS=1,5,20 \
  SUCC_UNIFIED_SELECTION_MODES=raw,finalizer \
  SUCC_UNIFIED_MOLEDIT_REFERENCE_CSV="$INPUT_CSV" \
  SUCC_UNIFIED_MOLEDIT_REQUIRE_TABLE1_COVERAGE=1 \
  bash "$SCRIPT_DIR/../unified_smiles_generator/run_unified_smiles_generator_benchmark_suite.sh"
else
  echo "=== Build official MuMO ADMET-AI + TDC candidate oracles ==="
  ORACLE_DIR="$OUTPUT_DIR/oracle"
  ORACLE_CSV="$ORACLE_DIR/generated_properties.csv"
  mkdir -p "$ORACLE_DIR"
  SUCC_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
  SUCC_ORACLE_INPUT_CSV="$OUTPUT_DIR/ranking/candidates.csv" \
  SUCC_ORACLE_OUTPUT_CSV="$ORACLE_CSV" \
  SUCC_ORACLE_WORK_DIR="$ORACLE_DIR/work" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="${SUCC_UCA_MUMO_MERGE_PROPERTIES_CSV:-$PROJECT_DIR/outputs/external_oracle_build_v1/generated_properties.csv}" \
  bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"

  echo "=== Score all n=20 MuMO candidates with the official evaluator ==="
  ALL_EVAL_DIR="$OUTPUT_DIR/official/all_n20"
  "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
    --prediction-csv "$OUTPUT_DIR/ranking/candidates.csv" \
    --output-dir "$ALL_EVAL_DIR" \
    --generated-properties-csv "$ORACLE_CSV" \
    --source-properties-csv "$ORACLE_CSV" \
    --group-column condition_id \
    --min-source-tanimoto 0.4 \
    --report-title "Common LLM GraphEditDSL MuMO any-at-20"

  for spec in "1 raw" "5 verifier" "20 verifier"; do
    read -r budget mode <<< "$spec"
    SELECTED_DIR="$OUTPUT_DIR/official/n${budget}_${mode}"
    SELECTED_CSV="$SELECTED_DIR/selected.csv"
    mkdir -p "$SELECTED_DIR"
    "$PYTHON_BIN" "$SCRIPT_DIR/select_external_verifier_prefix.py" \
      --detail-csv "$ALL_EVAL_DIR/external_multiproperty_detail.csv" \
      --output-csv "$SELECTED_CSV" \
      --budget "$budget" \
      --selection-mode "$mode" \
      --group-column condition_id
    "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
      --prediction-csv "$SELECTED_CSV" \
      --output-dir "$SELECTED_DIR/eval" \
      --generated-properties-csv "$ORACLE_CSV" \
      --source-properties-csv "$ORACLE_CSV" \
      --group-column condition_id \
      --min-source-tanimoto 0.4 \
      --report-title "Common LLM GraphEditDSL MuMO n=${budget} ${mode}"
  done
fi

echo "Common-LLM official $SUITE evaluation ready: $OUTPUT_DIR"
