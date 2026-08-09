#!/usr/bin/env bash
# Train the first common-LLM LoRA pilot on one small GPU allocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP_OVERLAY="${SUCC_UCA_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
OUTPUT_ROOT="${SUCC_UCA_COMMON_LLM_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_common_llm_pilot_v1}"
SFT_DIR="${SUCC_UCA_SFT_DIR:-$OUTPUT_ROOT/data/common_llm_sft}"
MODEL_DIR="${SUCC_UCA_MODEL_DIR:-$OUTPUT_ROOT/model/seed_${SUCC_UCA_SEED:-1701}}"
export PYTHONPATH="$DEP_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export TOKENIZERS_PARALLELISM=false

[[ -f "$SFT_DIR/train.jsonl" ]] || { echo "ERROR: missing train data: $SFT_DIR/train.jsonl" >&2; exit 2; }
[[ -d "$DEP_OVERLAY/peft" ]] || { echo "ERROR: missing PEFT overlay: $DEP_OVERLAY" >&2; exit 2; }
mkdir -p "$MODEL_DIR" "$HF_HOME"

"$PYTHON_BIN" "$SCRIPT_DIR/train_common_llm_lora.py" \
  --train-jsonl "$SFT_DIR/train.jsonl" \
  --validation-jsonl "$SFT_DIR/validation.jsonl" \
  --output-dir "$MODEL_DIR" \
  --base-model "${SUCC_UCA_BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --max-length "${SUCC_UCA_MAX_LENGTH:-512}" \
  --epochs "${SUCC_UCA_EPOCHS:-1}" \
  --batch-size "${SUCC_UCA_BATCH_SIZE:-2}" \
  --gradient-accumulation "${SUCC_UCA_GRADIENT_ACCUMULATION:-8}" \
  --learning-rate "${SUCC_UCA_LEARNING_RATE:-2e-4}" \
  --lora-r "${SUCC_UCA_LORA_R:-16}" \
  --lora-alpha "${SUCC_UCA_LORA_ALPHA:-32}" \
  --seed "${SUCC_UCA_SEED:-1701}"

echo "Common-LLM LoRA pilot ready: $MODEL_DIR"
