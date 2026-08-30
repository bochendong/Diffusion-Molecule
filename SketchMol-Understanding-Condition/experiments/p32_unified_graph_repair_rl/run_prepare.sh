#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P32_SCRIPT_DIR:?P32_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P32_OUTPUT_ROOT:-$PROJECT/outputs/p32_unified_graph_repair_rl/seed_32001}"
PY="${P32_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P32_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
P251="$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125"
P311="$PROJECT/outputs/p31_1_frontier_online_rloo/seed_31101"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/unified_constraint_agent:$PROJECT/experiments/p31_1_frontier_online_rloo:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUT/data"
"$PY" "$SCRIPT_DIR/prepare_records.py" \
  --train-jsonl "$P24/alignment_refresh/data/train.sft.jsonl" \
  --denovo-gate-jsonl "$P311/gate/data/prompts.jsonl" \
  --edit-gate-jsonl "$P251/data/gates/final.jsonl" \
  --base-model /scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct \
  --adapter-dir "$P24/alignment_refresh/model/adapter" \
  --output-dir "$OUT/data" \
  --batch-size 8 \
  --seed 32001
touch "$OUT/PREPARE_COMPLETE"
