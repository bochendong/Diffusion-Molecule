#!/usr/bin/env bash
# Train and gate the common-LLM RetrievedDeltaEdit planner at fixed final n=20.

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
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
V4_ROOT="${SUCC_UCA_V4_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_hierarchical_support_v4}"
V5_ROOT="${SUCC_UCA_V5_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_retrieved_delta_support_v5}"
SFT_ROOT="${SUCC_UCA_COMMON_LLM_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1}"
RUN_ROOT="${SUCC_UCA_DELTA_PLANNER_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_retrieved_delta_planner_v6}"
SEED="${SUCC_UCA_SEED:-1709}"
BASE_MODEL="${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
CANDIDATE_BUDGET="${SUCC_UCA_CANDIDATE_BUDGET:-20}"
PLANNER_CANDIDATE_LIMIT="${SUCC_UCA_PLANNER_CANDIDATE_LIMIT:-96}"

PROPOSER_TRAIN_ROWS="$V4_ROOT/data/proposer_train_rows.csv"
AUDIT_ROWS="$V4_ROOT/data/support_audit_disjoint_rows.csv"
FALLBACK_CANDIDATES="$V4_ROOT/graph_pool/benchmark_graph_edit_agent/graph_edit_agent_candidate_predictions.csv"
V5_BASELINE_SUMMARY="$V5_ROOT/support_gate_complete_v2/summary.json"
V5_COMPLETE_ORACLE="$V5_ROOT/oracle_complete_v2/generated_properties.csv"
BASE_ORACLE="$PROJECT_DIR/outputs/external_oracle_build_v1/generated_properties.csv"
SFT_DATA="$SFT_ROOT/data/common_llm_sft"
INPUT_ADAPTER="${SUCC_UCA_INPUT_ADAPTER:-$SFT_ROOT/model/seed_1703/adapter}"

PREFERENCE_DIR="$RUN_ROOT/data/seed_${SEED}"
MODEL_DIR="$RUN_ROOT/model/seed_${SEED}"
FORGETTING_DIR="$RUN_ROOT/anti_forgetting/seed_${SEED}"
POOL_DIR="$RUN_ROOT/candidate_pool"
HEURISTIC_CANDIDATES="$POOL_DIR/heuristic_top20.csv"
ENUMERATED_CANDIDATES="$POOL_DIR/enumerated_candidates.csv"
ENUMERATION_MANIFEST="$POOL_DIR/enumeration_manifest.json"
PLANNER_DIR="$RUN_ROOT/planner/seed_${SEED}"
PLANNER_CANDIDATES="$PLANNER_DIR/planner_top20.csv"
PLANNER_MANIFEST="$PLANNER_DIR/planner_manifest.json"
ORACLE_DIR="$RUN_ROOT/oracle/seed_${SEED}"
ORACLE_CSV="$ORACLE_DIR/generated_properties.csv"
OFFICIAL_DIR="$RUN_ROOT/benchmark_with_oracle/seed_${SEED}"
OFFICIAL_DETAIL="$OFFICIAL_DIR/external_multiproperty_detail.csv"
SUPPORT_DIR="$RUN_ROOT/support/seed_${SEED}"
GATE_DIR="$RUN_ROOT/gate/seed_${SEED}"

for path in \
  "$PYTHON_BIN" \
  "$ADMET_PYTHON_BIN" \
  "$PROPOSER_TRAIN_ROWS" \
  "$AUDIT_ROWS" \
  "$FALLBACK_CANDIDATES" \
  "$V5_BASELINE_SUMMARY" \
  "$V5_COMPLETE_ORACLE" \
  "$BASE_ORACLE" \
  "$SFT_DATA/train.jsonl" \
  "$SFT_DATA/validation.jsonl" \
  "$INPUT_ADAPTER/adapter_model.safetensors"; do
  [[ -e "$path" ]] || { echo "ERROR: missing v6 input: $path" >&2; exit 2; }
done
if [[ "$CANDIDATE_BUDGET" != "20" ]]; then
  echo "ERROR: v6 fixes the paper-facing candidate budget at n=20" >&2
  exit 2
fi
if (( PLANNER_CANDIDATE_LIMIT < CANDIDATE_BUDGET )); then
  echo "ERROR: planner candidate limit cannot be below n=20" >&2
  exit 2
fi

export PYTHONPATH="$DEP_OVERLAY:$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$PREFERENCE_DIR" "$MODEL_DIR" "$FORGETTING_DIR" "$POOL_DIR" "$PLANNER_DIR" "$ORACLE_DIR" "$OFFICIAL_DIR" "$SUPPORT_DIR" "$GATE_DIR"

echo "=== Validate frozen zero-overlap train/audit rows ==="
"$PYTHON_BIN" "$SCRIPT_DIR/audit_hierarchical_action_support.py" \
  --proposer-train-csv "$PROPOSER_TRAIN_ROWS" \
  --audit-rows-csv "$AUDIT_ROWS" \
  --validate-splits-only

if [[ ! -s "$PREFERENCE_DIR/train.jsonl" || ! -s "$PREFERENCE_DIR/validation.jsonl" ]]; then
  echo "=== Build paired train-only RetrievedDeltaEdit preferences ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/build_retrieved_delta_plan_preferences.py" \
    --train-csv "$PROPOSER_TRAIN_ROWS" \
    --output-dir "$PREFERENCE_DIR" \
    --validation-fraction "${SUCC_UCA_DELTA_VALIDATION_FRACTION:-0.10}" \
    --max-negatives-per-condition "${SUCC_UCA_MAX_DELTA_NEGATIVES:-1}" \
    --max-conditions-per-task "${SUCC_UCA_DELTA_TRAIN_CONDITIONS_PER_TASK:-50}" \
    --max-transforms-per-query "$PLANNER_CANDIDATE_LIMIT" \
    --seed "$SEED"
fi

if [[ ! -f "$MODEL_DIR/training_summary.json" || ! -f "$MODEL_DIR/adapter/adapter_model.safetensors" ]]; then
  echo "=== Preference-tune common LLM with balanced three-task replay ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/train_common_llm_preference.py" \
    --train-jsonl "$PREFERENCE_DIR/train.jsonl" \
    --validation-jsonl "$PREFERENCE_DIR/validation.jsonl" \
    --input-adapter-dir "$INPUT_ADAPTER" \
    --output-dir "$MODEL_DIR" \
    --base-model "$BASE_MODEL" \
    --max-length "${SUCC_UCA_DELTA_MAX_LENGTH:-1024}" \
    --epochs "${SUCC_UCA_DELTA_EPOCHS:-1}" \
    --batch-size "${SUCC_UCA_DELTA_BATCH_SIZE:-1}" \
    --gradient-accumulation "${SUCC_UCA_DELTA_GRADIENT_ACCUMULATION:-8}" \
    --learning-rate "${SUCC_UCA_DELTA_LR:-3e-6}" \
    --beta "${SUCC_UCA_DELTA_BETA:-2.0}" \
    --sft-weight "${SUCC_UCA_DELTA_SFT_WEIGHT:-0.10}" \
    --replay-jsonl "$SFT_DATA/train.jsonl" \
    --replay-sft-weight "${SUCC_UCA_REPLAY_SFT_WEIGHT:-0.10}" \
    --replay-batch-size 1 \
  --replay-max-per-origin "${SUCC_UCA_REPLAY_MAX_PER_ORIGIN:-256}" \
    --seed "$SEED"
fi

if [[ ! -f "$FORGETTING_DIR/baseline/summary.json" ]]; then
  echo "=== Re-evaluate stable common LLM on the frozen unified validation ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_pilot.py" \
    --input-jsonl "$SFT_DATA/validation.jsonl" \
    --output-dir "$FORGETTING_DIR/baseline" \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$INPUT_ADAPTER" \
    --variant stable_sft_seed_1703 \
    --batch-size 8 \
    --max-new-tokens 128
fi
if [[ ! -f "$FORGETTING_DIR/planner/summary.json" ]]; then
  echo "=== Check v6 common LLM for unified action-format forgetting ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_pilot.py" \
    --input-jsonl "$SFT_DATA/validation.jsonl" \
    --output-dir "$FORGETTING_DIR/planner" \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$MODEL_DIR/adapter" \
    --variant retrieved_delta_planner_v6 \
    --batch-size 8 \
    --max-new-tokens 128
fi

if [[ ! -f "$ENUMERATED_CANDIDATES" || ! -f "$ENUMERATION_MANIFEST" ]]; then
  echo "=== Enumerate the existing train-only delta action space once ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/build_retrieved_delta_edit_candidates.py" \
    --train-csv "$PROPOSER_TRAIN_ROWS" \
    --eval-csv "$AUDIT_ROWS" \
    --fallback-candidates-csv "$FALLBACK_CANDIDATES" \
    --output-csv "$HEURISTIC_CANDIDATES" \
    --enumerated-output-csv "$ENUMERATED_CANDIDATES" \
    --manifest-json "$ENUMERATION_MANIFEST" \
    --candidate-budget 20 \
    --min-retrieval-similarity 0.15 \
    --max-transforms-per-query "$PLANNER_CANDIDATE_LIMIT" \
    --min-core-heavy-atoms 5 \
    --max-variable-heavy-atoms 30 \
    --min-source-tanimoto 0.4
fi

if [[ ! -f "$PLANNER_CANDIDATES" || ! -f "$PLANNER_MANIFEST" ]]; then
  echo "=== Let the common LLM select exactly 20 candidates per audit condition ==="
  "$PYTHON_BIN" "$SCRIPT_DIR/rank_retrieved_delta_candidates.py" \
    --enumerated-candidates-csv "$ENUMERATED_CANDIDATES" \
    --output-csv "$PLANNER_CANDIDATES" \
    --manifest-json "$PLANNER_MANIFEST" \
    --output-dir "$PLANNER_DIR" \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$MODEL_DIR/adapter" \
    --preference-manifest-json "$PREFERENCE_DIR/manifest.json" \
    --candidate-budget 20 \
    --planner-candidate-limit "$PLANNER_CANDIDATE_LIMIT" \
    --min-source-tanimoto 0.4 \
    --score-batch-size "${SUCC_UCA_DELTA_SCORE_BATCH_SIZE:-8}" \
    --max-length "${SUCC_UCA_DELTA_MAX_LENGTH:-1024}"
fi

if [[ ! -f "$ORACLE_CSV" ]]; then
  echo "=== Run the complete official oracle on only the selected n=20 pool ==="
  SUCC_PYTHON_BIN="$PYTHON_BIN" \
  SUCC_ADMET_PYTHON_BIN="$ADMET_PYTHON_BIN" \
  SUCC_ORACLE_INPUT_CSV="$PLANNER_CANDIDATES" \
  SUCC_ORACLE_OUTPUT_CSV="$ORACLE_CSV" \
  SUCC_ORACLE_WORK_DIR="$ORACLE_DIR/work" \
  SUCC_ORACLE_MERGE_PROPERTIES_CSV="$V5_COMPLETE_ORACLE,$BASE_ORACLE" \
  SUCC_ORACLE_ADMET_REQUIRED_PROPERTIES="bbbp,hia,mutagenicity" \
  bash "$CODE_PROJECT_DIR/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
fi

"$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); m=p["missing_counts"]; assert m.get("drd2", 1) == 0, m' \
  "${ORACLE_CSV%.csv}.summary.json"

if [[ ! -f "$OFFICIAL_DETAIL" ]]; then
  echo "=== Materialize complete fixed-n official metrics ==="
  "$PYTHON_BIN" "$CODE_PROJECT_DIR/scripts/evaluate_external_multiproperty_predictions.py" \
    --prediction-csv "$PLANNER_CANDIDATES" \
    --output-dir "$OFFICIAL_DIR" \
    --generated-properties-csv "$ORACLE_CSV" \
    --source-properties-csv "$ORACLE_CSV" \
    --group-column condition_id \
    --min-source-tanimoto 0.4 \
    --report-title "Common-LLM RetrievedDelta planner v6 train-only audit n=20"
fi

echo "=== Build support summary and compare against frozen v5 ==="
"$PYTHON_BIN" "$SCRIPT_DIR/audit_hierarchical_action_support.py" \
  --proposer-train-csv "$PROPOSER_TRAIN_ROWS" \
  --audit-rows-csv "$AUDIT_ROWS" \
  --official-detail-csv "$OFFICIAL_DETAIL" \
  --candidate-manifest-json "$PLANNER_MANIFEST" \
  --output-dir "$SUPPORT_DIR" \
  --candidate-budget 20 \
  --protocol hierarchical_common_agent_retrieved_delta_planner_v6 \
  --proposal-budget 0 \
  --method-label "common LLM selection over train-only RetrievedDeltaEdit actions" \
  --min-property-any-rate 0.46 \
  --min-strict-any-rate 0.46 \
  --min-full-oracle-condition-rate 1.0

"$PYTHON_BIN" "$SCRIPT_DIR/finalize_retrieved_delta_planner_gate.py" \
  --support-summary "$SUPPORT_DIR/summary.json" \
  --baseline-support-summary "$V5_BASELINE_SUMMARY" \
  --baseline-format-summary "$FORGETTING_DIR/baseline/summary.json" \
  --candidate-format-summary "$FORGETTING_DIR/planner/summary.json" \
  --preference-manifest "$PREFERENCE_DIR/manifest.json" \
  --training-summary "$MODEL_DIR/training_summary.json" \
  --output-dir "$GATE_DIR" \
  --min-support-rate 0.46 \
  --min-support-gain 0.06 \
  --max-overall-forgetting-drop 0.02 \
  --max-origin-forgetting-drop 0.05

echo "Common-LLM RetrievedDelta planner v6 ready: $GATE_DIR/report.md"
