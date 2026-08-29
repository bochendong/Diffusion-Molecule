#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P31_SCRIPT_DIR:?P31_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P31_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P31_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
INPUT="${P31_TRAIN_JSONL:-$P24/alignment_refresh/data/train.sft.jsonl}"
OUT="${P31_OUTPUT_ROOT:-$PROJECT/outputs/p31_reward_support_audit/seed_31001}"
test -f "$P24/alignment_refresh/ALIGNMENT_REFRESH_COMPLETE"
test -f "$INPUT"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="$DEP:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$SCRIPT_DIR/build_audit_prompts.py" \
  --input-jsonl "$INPUT" --output-dir "$OUT/data" \
  --manifest "$OUT/data/manifest.json" --per-bucket 60 --shards 4 --seed 31001
touch "$OUT/PREPARED"
