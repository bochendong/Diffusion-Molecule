#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P33_SCRIPT_DIR:?P33_SCRIPT_DIR must be exported}"
ARM="${P33_ARM:?P33_ARM must be joint, denovo, or edit}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P33_OUTPUT_ROOT:-$PROJECT/outputs/p33_joint_vs_separate_pilot/seed_33001}"
PY="${P33_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P33_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P33_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
case "$ARM" in joint|denovo|edit) ;; *) echo "ERROR: invalid P33_ARM=$ARM" >&2; exit 2 ;; esac
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"$PY" "$SCRIPT_DIR/train_arm.py" --train-jsonl "$OUT/data/train.$ARM.jsonl" \
  --output-dir "$OUT/model/$ARM" --base-model "$BASE" --arm "$ARM" \
  --epochs 1 --gradient-accumulation 32 --learning-rate 8e-5 --seed 33001
