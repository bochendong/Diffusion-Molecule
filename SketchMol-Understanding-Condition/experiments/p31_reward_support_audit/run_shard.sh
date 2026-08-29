#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P31_SCRIPT_DIR:?P31_SCRIPT_DIR must be exported}"
SHARD="${P31_SHARD:?P31_SHARD must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P31_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P31_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P31_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
ADAPTER="${P31_ADAPTER:-$P24/alignment_refresh/model/adapter}"
OUT="${P31_OUTPUT_ROOT:-$PROJECT/outputs/p31_reward_support_audit/seed_31001}"
test -f "$OUT/PREPARED"
test -f "$ADAPTER/adapter_model.safetensors"
input=$(printf '%s/data/shard-%02d.jsonl' "$OUT" "$SHARD")
output=$(printf '%s/shards/shard-%02d.candidates.jsonl' "$OUT" "$SHARD")
summary=$(printf '%s/shards/shard-%02d.summary.json' "$OUT" "$SHARD")
test -f "$input"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/generate_score_shard.py" \
  --input-jsonl "$input" --output-jsonl "$output" --summary-json "$summary" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" --shard "$SHARD" \
  --sample-count 16 --batch-size 4 --seed 31001
touch "$(printf '%s/SHARD_%02d_COMPLETE' "$OUT" "$SHARD")"
