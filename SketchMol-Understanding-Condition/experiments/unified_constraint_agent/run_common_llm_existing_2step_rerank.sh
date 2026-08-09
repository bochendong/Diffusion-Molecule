#!/usr/bin/env bash
# Rerank the completed official MuMO 2-step top-20 pool with seed_1705.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
SOURCE_ROOT="${SUCC_UCA_2STEP_SOURCE_ROOT:-$PROJECT_DIR/outputs/external_mumo_official_graph_edit_heuristic_2step_v1}"
OFFICIAL_DETAIL="${SUCC_UCA_2STEP_OFFICIAL_DETAIL:-$SOURCE_ROOT/benchmark_with_oracle_v1/external_multiproperty_detail.csv}"
PLAN_JSONL="${SUCC_UCA_2STEP_PLAN_JSONL:-$SOURCE_ROOT/graph_edit_plans.jsonl}"
PREF_ROOT="${SUCC_UCA_VERIFIER_PREFERENCE_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_verifier_preference_v2}"
ADAPTER_DIR="${SUCC_UCA_2STEP_ADAPTER:-$PREF_ROOT/model/seed_1705/adapter}"
OUTPUT_DIR="${SUCC_UCA_2STEP_OUTPUT_DIR:-$PROJECT_DIR/outputs/unified_constraint_agent_existing_2step_rerank_v1}"
BASE_MODEL="${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

export PYTHONPATH="$DEP_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false

for path in "$OFFICIAL_DETAIL" "$PLAN_JSONL" "$ADAPTER_DIR/adapter_model.safetensors"; do
  [[ -f "$path" ]] || { echo "ERROR: missing existing-2step input: $path" >&2; exit 2; }
done
mkdir -p "$OUTPUT_DIR"

args=(
  "$SCRIPT_DIR/rerank_common_llm_existing_action_plans.py"
  --official-detail-csv "$OFFICIAL_DETAIL"
  --plan-jsonl "$PLAN_JSONL"
  --output-dir "$OUTPUT_DIR"
  --base-model "$BASE_MODEL"
  --adapter-dir "$ADAPTER_DIR"
  --candidate-budget 20
  --verifier-k 5
  --score-batch-size "${SUCC_UCA_SCORE_BATCH_SIZE:-4}"
  --max-length "${SUCC_UCA_EVAL_MAX_LENGTH:-1024}"
)
if [[ -n "${SUCC_UCA_2STEP_MAX_ROWS:-}" ]]; then
  args+=(--max-rows "$SUCC_UCA_2STEP_MAX_ROWS")
fi

echo "Common-LLM existing 2-step rerank"
echo "  official_detail=$OFFICIAL_DETAIL"
echo "  plan_jsonl=$PLAN_JSONL"
echo "  adapter_dir=$ADAPTER_DIR"
echo "  output_dir=$OUTPUT_DIR"
"$PYTHON_BIN" "${args[@]}"

echo "Common-LLM existing 2-step rerank ready: $OUTPUT_DIR/report.md"
