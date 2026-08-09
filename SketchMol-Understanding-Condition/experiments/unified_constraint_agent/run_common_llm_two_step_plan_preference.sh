#!/usr/bin/env bash
# Build train-only MuMO 2-step preferences, tune with replay, gate, and optionally run formal test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
SFT_ROOT="${SUCC_UCA_COMMON_LLM_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1}"
RUN_ROOT="${SUCC_UCA_PLAN_PREFERENCE_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_two_step_plan_preference_v3}"
TRAIN_POOL="$RUN_ROOT/train_pool"
PREF_DATA="$RUN_ROOT/data/seed_${SUCC_UCA_SEED:-1706}"
MODEL_DIR="$RUN_ROOT/model/seed_${SUCC_UCA_SEED:-1706}"
VALIDATION_ROOT="$RUN_ROOT/validation/seed_${SUCC_UCA_SEED:-1706}"
FORMAL_ROOT="$RUN_ROOT/formal_test/seed_${SUCC_UCA_SEED:-1706}"
SFT_DATA="$SFT_ROOT/data/common_llm_sft"
INPUT_ADAPTER="${SUCC_UCA_INPUT_ADAPTER:-$SFT_ROOT/model/seed_1703/adapter}"
BASE_MODEL="${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MUMO_TRAIN_JSON="${SUCC_UCA_MUMO_TRAIN_JSON:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/train.json}"
TRAIN_DETAIL="$TRAIN_POOL/benchmark_graph_edit_agent/external_multiproperty_detail.csv"
TRAIN_PLANS="$TRAIN_POOL/graph_edit_plans.jsonl"
FORMAL_SOURCE_ROOT="${SUCC_UCA_FORMAL_SOURCE_ROOT:-$PROJECT_DIR/outputs/external_mumo_official_graph_edit_heuristic_2step_v1}"
FORMAL_DETAIL="${SUCC_UCA_FORMAL_DETAIL:-$FORMAL_SOURCE_ROOT/benchmark_with_oracle_v1/external_multiproperty_detail.csv}"
FORMAL_PLANS="${SUCC_UCA_FORMAL_PLANS:-$FORMAL_SOURCE_ROOT/graph_edit_plans.jsonl}"
FORMAL_RECONSTRUCTED="${SUCC_UCA_FORMAL_RECONSTRUCTED_PLANS:-$PROJECT_DIR/outputs/unified_constraint_agent_existing_2step_rerank_v1/reconstructed_candidate_plans.jsonl}"
SEED="${SUCC_UCA_SEED:-1706}"

export PYTHONPATH="$DEP_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false

for path in \
  "$PYTHON_BIN" \
  "$MUMO_TRAIN_JSON" \
  "$SFT_DATA/train.jsonl" \
  "$INPUT_ADAPTER/adapter_model.safetensors" \
  "$FORMAL_DETAIL" \
  "$FORMAL_PLANS"; do
  [[ -e "$path" ]] || { echo "ERROR: missing two-step plan preference input: $path" >&2; exit 2; }
done
mkdir -p "$RUN_ROOT" "$PREF_DATA" "$MODEL_DIR" "$VALIDATION_ROOT" "$FORMAL_ROOT"

if [[ ! -f "$TRAIN_DETAIL" || ! -f "$TRAIN_PLANS" ]]; then
  echo "=== Build and officially score train-only MuMO 2-step n=20 pool ==="
  export SUCC_PYTHON_BIN="$PYTHON_BIN"
  export SUCC_EXTERNAL_GRAPH_EDIT_SOURCE_FILE="$MUMO_TRAIN_JSON"
  export SUCC_EXTERNAL_GRAPH_EDIT_OUTPUT_DIR="$TRAIN_POOL"
  export SUCC_EXTERNAL_GRAPH_EDIT_SUITE="mumo"
  export SUCC_EXTERNAL_GRAPH_EDIT_TASK_SPLIT="all"
  export SUCC_EXTERNAL_GRAPH_EDIT_INPUT_SPLIT="train"
  export SUCC_EXTERNAL_GRAPH_EDIT_MAX_ROWS_PER_TASK="${SUCC_UCA_MUMO_TRAIN_ROWS_PER_TASK:-50}"
  export SUCC_EXTERNAL_GRAPH_EDIT_FORCE_EXPORT="${SUCC_UCA_FORCE_TRAIN_POOL_EXPORT:-0}"
  export SUCC_EXTERNAL_GRAPH_EDIT_REQUIRE_DIRECT_PROPOSALS="0"
  export SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_MODE="heuristic_graph_dsl"
  export SUCC_EXTERNAL_GRAPH_EDIT_SELECTION_MODE="similarity_first"
  export SUCC_EXTERNAL_GRAPH_EDIT_PLANNER_STEPS="2"
  export SUCC_EXTERNAL_GRAPH_EDIT_BEAM_SIZE="${SUCC_UCA_TRAIN_POOL_BEAM_SIZE:-128}"
  export SUCC_EXTERNAL_GRAPH_EDIT_SITE_LIMIT="${SUCC_UCA_TRAIN_POOL_SITE_LIMIT:-48}"
  export SUCC_EXTERNAL_GRAPH_EDIT_MAX_PLANS_PER_PROPERTY="${SUCC_UCA_TRAIN_POOL_MAX_PLANS_PER_PROPERTY:-192}"
  export SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_PARENT="${SUCC_UCA_TRAIN_POOL_MAX_CANDIDATES_PER_PARENT:-192}"
  export SUCC_EXTERNAL_GRAPH_EDIT_MAX_CANDIDATES_PER_ROW="${SUCC_UCA_TRAIN_POOL_MAX_CANDIDATES_PER_ROW:-4096}"
  export SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_FIRST_MIN_LOCAL_SUCCESS_FRACTION="0.9"
  export SUCC_EXTERNAL_GRAPH_EDIT_PROPERTY_WEIGHT="80"
  export SUCC_EXTERNAL_GRAPH_EDIT_DISTANCE_WEIGHT="8"
  export SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_WEIGHT="80"
  export SUCC_EXTERNAL_GRAPH_EDIT_SIMILARITY_BONUS="180"
  export SUCC_EXTERNAL_GRAPH_EDIT_COPY_PENALTY="10"
  export SUCC_EXTERNAL_GRAPH_EDIT_TOP_K_CANDIDATES="20"
  export SUCC_EXTERNAL_GRAPH_EDIT_BUILD_ORACLE_CSV="1"
  export SUCC_EXTERNAL_GRAPH_EDIT_CHECKPOINT="1"
  export SUCC_EXTERNAL_GRAPH_EDIT_RESUME="1"
  bash SketchMol-Understanding-Condition/scripts/run_direct_smiles_external_mumo_graph_edit_agent.sh
fi

for path in "$TRAIN_DETAIL" "$TRAIN_PLANS"; do
  [[ -f "$path" ]] || { echo "ERROR: train-only two-step pool did not produce $path" >&2; exit 2; }
done

echo "=== Build strict-positive two-step plan preferences with similarity hard negatives ==="
"$PYTHON_BIN" "$SCRIPT_DIR/build_common_llm_plan_preferences.py" \
  --official-detail-csv "$TRAIN_DETAIL" \
  --plan-jsonl "$TRAIN_PLANS" \
  --output-dir "$PREF_DATA" \
  --candidate-budget 20 \
  --max-negatives-per-condition "${SUCC_UCA_MAX_PLAN_NEGATIVES:-3}" \
  --validation-fraction "${SUCC_UCA_PLAN_VALIDATION_FRACTION:-0.10}" \
  --seed "$SEED" \
  --require-input-split train

echo "=== Train joint-plan preference adapter with task-balanced SFT replay ==="
"$PYTHON_BIN" "$SCRIPT_DIR/train_common_llm_preference.py" \
  --train-jsonl "$PREF_DATA/train.jsonl" \
  --validation-jsonl "$PREF_DATA/validation.jsonl" \
  --input-adapter-dir "$INPUT_ADAPTER" \
  --output-dir "$MODEL_DIR" \
  --base-model "$BASE_MODEL" \
  --max-length "${SUCC_UCA_PLAN_MAX_LENGTH:-1024}" \
  --epochs "${SUCC_UCA_PLAN_EPOCHS:-1}" \
  --batch-size "${SUCC_UCA_PLAN_BATCH_SIZE:-1}" \
  --gradient-accumulation "${SUCC_UCA_PLAN_GRADIENT_ACCUMULATION:-8}" \
  --learning-rate "${SUCC_UCA_PLAN_LR:-3e-6}" \
  --beta "${SUCC_UCA_PLAN_BETA:-2.0}" \
  --sft-weight "${SUCC_UCA_PLAN_SFT_WEIGHT:-0.10}" \
  --replay-jsonl "$SFT_DATA/train.jsonl" \
  --replay-sft-weight "${SUCC_UCA_REPLAY_SFT_WEIGHT:-0.10}" \
  --replay-batch-size "${SUCC_UCA_REPLAY_BATCH_SIZE:-1}" \
  --replay-max-per-origin "${SUCC_UCA_REPLAY_MAX_PER_ORIGIN:-256}" \
  --seed "$SEED"

rerank_validation() {
  local adapter_dir="$1"
  local output_dir="$2"
  local variant="$3"
  "$PYTHON_BIN" "$SCRIPT_DIR/rerank_common_llm_existing_action_plans.py" \
    --official-detail-csv "$TRAIN_DETAIL" \
    --plan-jsonl "$TRAIN_PLANS" \
    --condition-ids-file "$PREF_DATA/validation_condition_ids.txt" \
    --reconstructed-plans-jsonl "$PREF_DATA/reconstructed_candidate_plans.jsonl" \
    --output-dir "$output_dir" \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$adapter_dir" \
    --candidate-budget 20 \
    --verifier-k 5 \
    --score-batch-size "${SUCC_UCA_SCORE_BATCH_SIZE:-4}" \
    --max-length "${SUCC_UCA_PLAN_MAX_LENGTH:-1024}" \
    --plan-score-mode joint_plan_logprob \
    --variant "$variant"
}

echo "=== Held-out train-only validation: stable SFT vs plan preference ==="
rerank_validation "$INPUT_ADAPTER" "$VALIDATION_ROOT/sft" "common_llm_sft_joint_plan"
rerank_validation "$MODEL_DIR/adapter" "$VALIDATION_ROOT/plan_preference" "common_llm_two_step_plan_preference_v3"

set +e
"$PYTHON_BIN" "$SCRIPT_DIR/compare_common_llm_plan_rankers.py" \
  --baseline-summary "$VALIDATION_ROOT/sft/summary.json" \
  --candidate-summary "$VALIDATION_ROOT/plan_preference/summary.json" \
  --output-json "$VALIDATION_ROOT/gate.json" \
  --output-report "$VALIDATION_ROOT/gate.md" \
  --verifier-k 5 \
  --min-primary-gain "${SUCC_UCA_MIN_PRIMARY_GAIN:-0.01}" \
  --max-source-similarity-drop "${SUCC_UCA_MAX_SIMILARITY_DROP:-0.01}" \
  --fail-on-stop
gate_status=$?
set -e
if [[ "$gate_status" -ne 0 ]]; then
  echo "Plan preference gate stopped formal test; see $VALIDATION_ROOT/gate.md"
  exit 0
fi

if [[ "${SUCC_UCA_RUN_FORMAL_AFTER_GATE:-1}" == "1" ]]; then
  echo "=== Formal MuMO test rerank after held-out go decision ==="
  formal_plan_args=()
  if [[ -f "$FORMAL_RECONSTRUCTED" ]]; then
    formal_plan_args+=(--reconstructed-plans-jsonl "$FORMAL_RECONSTRUCTED")
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/rerank_common_llm_existing_action_plans.py" \
    --official-detail-csv "$FORMAL_DETAIL" \
    --plan-jsonl "$FORMAL_PLANS" \
    --output-dir "$FORMAL_ROOT" \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$MODEL_DIR/adapter" \
    --candidate-budget 20 \
    --verifier-k 5 \
    --score-batch-size "${SUCC_UCA_SCORE_BATCH_SIZE:-4}" \
    --max-length "${SUCC_UCA_PLAN_MAX_LENGTH:-1024}" \
    --plan-score-mode joint_plan_logprob \
    --variant common_llm_two_step_plan_preference_v3 \
    "${formal_plan_args[@]}"
fi

echo "Common-LLM two-step plan preference v3 ready: $RUN_ROOT"
