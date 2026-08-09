#!/usr/bin/env bash
# Compare base and tuned LLM ranking over executable n=20 GraphEditDSL pools.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
OUTPUT_ROOT="${SUCC_UCA_COMMON_LLM_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1}"
SEED="${SUCC_UCA_SEED:-1703}"
SFT_DIR="${SUCC_UCA_SFT_DIR:-$OUTPUT_ROOT/data/common_llm_sft}"
ADAPTER_DIR="${SUCC_UCA_ADAPTER_DIR:-$OUTPUT_ROOT/model/seed_${SEED}/adapter}"
EVAL_DIR="${SUCC_UCA_CONSTRAINED_EVAL_DIR:-$OUTPUT_ROOT/constrained_eval/seed_${SEED}}"
BASE_MODEL="${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export PYTHONPATH="$DEP_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false

[[ -f "$SFT_DIR/validation.jsonl" ]] || { echo "ERROR: missing validation data" >&2; exit 2; }
[[ -f "$ADAPTER_DIR/adapter_model.safetensors" ]] || { echo "ERROR: missing adapter" >&2; exit 2; }

common_args=(
  --input-jsonl "$SFT_DIR/validation.jsonl"
  --base-model "$BASE_MODEL"
  --candidate-budget 20
  --site-limit "${SUCC_UCA_SITE_LIMIT:-32}"
  --score-batch-size "${SUCC_UCA_SCORE_BATCH_SIZE:-4}"
  --max-length "${SUCC_UCA_MAX_LENGTH:-512}"
)

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_constrained_actions.py" \
  "${common_args[@]}" \
  --output-dir "$EVAL_DIR/base" \
  --variant base

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_constrained_actions.py" \
  "${common_args[@]}" \
  --output-dir "$EVAL_DIR/tuned" \
  --adapter-dir "$ADAPTER_DIR" \
  --variant tuned

echo "Common-LLM constrained n=20 evaluation ready: $EVAL_DIR"
