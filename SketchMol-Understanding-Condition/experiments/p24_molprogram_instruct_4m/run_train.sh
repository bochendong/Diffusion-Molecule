#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P24_SCRIPT_DIR:?P24_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE="${P24_TRAIN_MODE:?P24_TRAIN_MODE must be gate or full}"
WORK_ROOT="${P24_WORK_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule/processed/molprogram-instruct-4m-v1}"
RELEASE="${P24_RELEASE_ROOT:-$WORK_ROOT/release}"
BASE="${P24_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
INPUT_ADAPTER="${P24_INPUT_ADAPTER:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned/model/stage1_v2/adapter}"
OUTPUT_ROOT="${P24_OUTPUT_ROOT:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003}"
PY="${P24_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P24_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
fi
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ "$MODE" == gate ]]; then
  steps="${P24_MAX_STEPS:-500}"
  # 500 x 26 = 13,000 examples = 1,000 per each of 13 tasks.
  batch_size="${P24_BATCH_SIZE:-1}"
  accumulation="${P24_GRADIENT_ACCUMULATION:-26}"
  # Keep the corrected gate separate from the superseded 16k-example gate so
  # checkpoint auto-resume cannot silently preserve the wrong contract.
  output="$OUTPUT_ROOT/gate_13k"
elif [[ "$MODE" == full ]]; then
  # 16,283 x 5 x 13 = 1,058,395 examples = 81,415 per each of 13 tasks.
  steps="${P24_MAX_STEPS:-16283}"
  batch_size="${P24_BATCH_SIZE:-5}"
  accumulation="${P24_GRADIENT_ACCUMULATION:-13}"
  output="$OUTPUT_ROOT/full"
else
  echo "ERROR: P24_TRAIN_MODE must be gate or full" >&2
  exit 2
fi
mkdir -p "$output"
resume=()
compgen -G "$output/checkpoint-*" >/dev/null && resume+=(--resume-from-checkpoint)
"$PY" "$SCRIPT_DIR/train_indexed_sft.py" \
  --release-root "$RELEASE" --output-dir "$output" --base-model "$BASE" \
  --input-adapter "$INPUT_ADAPTER" --max-steps "$steps" \
  --per-device-batch-size "$batch_size" \
  --gradient-accumulation "$accumulation" --learning-rate "${P24_LEARNING_RATE:-1e-5}" \
  --save-steps "${P24_SAVE_STEPS:-500}" --seed 24003 "${resume[@]}"
echo "P24 $MODE training complete: $output"
