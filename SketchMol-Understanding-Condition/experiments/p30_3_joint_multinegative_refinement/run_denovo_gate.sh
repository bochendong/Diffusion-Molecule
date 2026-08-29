#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P303_SCRIPT_DIR:?P303_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P303_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P303_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P303_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P301="$PROJECT/outputs/p30_1_alignment_raw1_rl/seed_30101"
OUT="${P303_OUTPUT_ROOT:-$PROJECT/outputs/p30_3_joint_multinegative_refinement/seed_30301}"
ADAPTER="$OUT/model/refined/adapter"
test -f "$OUT/TRAIN_COMPLETE"; test -f "$ADAPTER/adapter_model.safetensors"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$PROJECT/experiments/p30_1_alignment_raw1_rl/evaluate_small_raw1_gate.py" \
  --prompts-jsonl "$P301/small_gate/data/prompts.jsonl" \
  --baseline-summary "$P301/small_gate/data/baseline_summary.json" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" --output-dir "$OUT/gate/denovo" \
  --batch-size 8 --macro-delta-min 0.02 --valid-delta-min -0.01 --bucket-delta-min -0.10
touch "$OUT/DENOVO_GATE_COMPLETE"
