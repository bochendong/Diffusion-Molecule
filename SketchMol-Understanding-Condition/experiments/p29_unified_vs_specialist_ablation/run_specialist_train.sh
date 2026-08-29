#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P29_SCRIPT_DIR:?P29_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P24="$PROJECT/experiments/p24_molprogram_instruct_4m"
MODE="${P29_SPECIALIST_MODE:?P29_SPECIALIST_MODE must be construction or editing}"
WORK_ROOT="${P24_WORK_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule/processed/molprogram-instruct-4m-v1}"
RELEASE="${P24_RELEASE_ROOT:-$WORK_ROOT/release}"
BASE="${P24_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
INPUT_ADAPTER="${P29_INPUT_ADAPTER:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned/model/stage1_v2/adapter}"
OUTPUT_ROOT="${P29_OUTPUT_ROOT:-$PROJECT/outputs/p29_unified_vs_specialist_ablation/seed_24003}"
PY="${P24_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P24_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
fi
export PYTHONPATH="$DEP:$P24:$PROJECT/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

test -f "$RELEASE/RELEASE_COMPLETE"
test -f "$INPUT_ADAPTER/adapter_model.safetensors"

case "$MODE" in
  construction)
    task_mode="de_novo"
    output="$OUTPUT_ROOT/construction_specialist"
    expected_examples=488490
    ;;
  editing)
    task_mode="edit"
    output="$OUTPUT_ROOT/editing_specialist"
    expected_examples=569905
    ;;
  *)
    echo "ERROR: P29_SPECIALIST_MODE must be construction or editing" >&2
    exit 2
    ;;
esac

mkdir -p "$output"
resume=()
compgen -G "$output/checkpoint-*" >/dev/null && resume+=(--resume-from-checkpoint)
"$PY" "$SCRIPT_DIR/train_specialist_sft.py" \
  --release-root "$RELEASE" --output-dir "$output" --base-model "$BASE" \
  --input-adapter "$INPUT_ADAPTER" --task-mode "$task_mode" \
  --rows-per-task 81415 --expected-release-rows 2569919 \
  --expected-examples "$expected_examples" \
  --per-device-batch-size 1 --gradient-accumulation 65 \
  --learning-rate 1e-5 --warmup-steps 100 --save-steps 1000 \
  --logging-steps 10 --seed 24003 "${resume[@]}"

test -f "$output/TRAINING_COMPLETE"
python3 - "$output/training_summary.json" "$expected_examples" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
if summary["effective_examples"] != expected:
    raise SystemExit(
        f"specialist exposure mismatch: {summary['effective_examples']} != {expected}"
    )
PY
echo "P29 $MODE specialist training complete: $output"
