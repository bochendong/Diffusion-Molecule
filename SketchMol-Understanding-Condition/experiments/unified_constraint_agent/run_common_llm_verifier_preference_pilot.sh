#!/usr/bin/env bash
# Train verifier-aligned preference v2 and compare it with stable SFT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
SFT_ROOT="${SUCC_UCA_COMMON_LLM_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1}"
PREF_ROOT="${SUCC_UCA_VERIFIER_PREFERENCE_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_verifier_preference_v2}"
INPUT_SEED="${SUCC_UCA_INPUT_SEED:-1703}"
OUTPUT_SEED="${SUCC_UCA_SEED:-1705}"
SFT_DATA="$SFT_ROOT/data/common_llm_sft"
INPUT_ADAPTER="${SUCC_UCA_INPUT_ADAPTER:-$SFT_ROOT/model/seed_${INPUT_SEED}/adapter}"
PREF_DATA="$PREF_ROOT/data/seed_${OUTPUT_SEED}"
MODEL_DIR="$PREF_ROOT/model/seed_${OUTPUT_SEED}"
EVAL_DIR="$PREF_ROOT/eval/seed_${OUTPUT_SEED}"
BASE_MODEL="${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export PYTHONPATH="$DEP_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false

for path in "$SFT_DATA/train.jsonl" "$SFT_DATA/validation.jsonl" "$INPUT_ADAPTER/adapter_model.safetensors"; do
  [[ -f "$path" ]] || { echo "ERROR: missing verifier-preference input: $path" >&2; exit 2; }
done

if [[ "${SUCC_UCA_SKIP_VERIFIER_DATA:-0}" != "1" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/build_common_llm_verifier_preferences.py" \
    --train-jsonl "$SFT_DATA/train.jsonl" \
    --validation-jsonl "$SFT_DATA/validation.jsonl" \
    --output-dir "$PREF_DATA" \
    --candidate-budget 20 \
    --enumeration-attempt-budget "${SUCC_UCA_ENUMERATION_ATTEMPT_BUDGET:-64}" \
    --site-limit "${SUCC_UCA_SITE_LIMIT:-32}" \
    --negatives-per-example "${SUCC_UCA_NEGATIVES_PER_EXAMPLE:-2}"
fi

if [[ "${SUCC_UCA_SKIP_VERIFIER_TRAIN:-0}" != "1" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/train_common_llm_preference.py" \
    --train-jsonl "$PREF_DATA/train.jsonl" \
    --validation-jsonl "$PREF_DATA/validation.jsonl" \
    --input-adapter-dir "$INPUT_ADAPTER" \
    --output-dir "$MODEL_DIR" \
    --base-model "$BASE_MODEL" \
    --max-length "${SUCC_UCA_PREFERENCE_MAX_LENGTH:-512}" \
    --epochs "${SUCC_UCA_PREFERENCE_EPOCHS:-1}" \
    --batch-size "${SUCC_UCA_PREFERENCE_BATCH_SIZE:-1}" \
    --gradient-accumulation "${SUCC_UCA_PREFERENCE_GRADIENT_ACCUMULATION:-8}" \
    --learning-rate "${SUCC_UCA_PREFERENCE_LR:-5e-6}" \
    --beta "${SUCC_UCA_PREFERENCE_BETA:-2.0}" \
    --sft-weight "${SUCC_UCA_PREFERENCE_SFT_WEIGHT:-0.10}" \
    --seed "$OUTPUT_SEED"
fi

common_eval_args=(
  --input-jsonl "$SFT_DATA/validation.jsonl"
  --base-model "$BASE_MODEL"
  --candidate-budget 20
  --enumeration-attempt-budget "${SUCC_UCA_ENUMERATION_ATTEMPT_BUDGET:-64}"
  --site-limit "${SUCC_UCA_SITE_LIMIT:-32}"
  --score-batch-size "${SUCC_UCA_SCORE_BATCH_SIZE:-4}"
  --max-length "${SUCC_UCA_EVAL_MAX_LENGTH:-1024}"
)

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_constrained_actions.py" \
  "${common_eval_args[@]}" \
  --output-dir "$EVAL_DIR/sft" \
  --adapter-dir "$INPUT_ADAPTER" \
  --variant sft

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_common_llm_constrained_actions.py" \
  "${common_eval_args[@]}" \
  --output-dir "$EVAL_DIR/verifier_preference" \
  --adapter-dir "$MODEL_DIR/adapter" \
  --variant verifier_preference

echo "Common-LLM verifier preference v2 ready: $PREF_ROOT"
