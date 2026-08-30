#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P34_SCRIPT_DIR:?P34_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P34_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P34_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P34_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
P29="$PROJECT/outputs/p29_unified_vs_specialist_ablation/seed_24003"
P251="$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125"
P324="$PROJECT/outputs/p32_4_edit_specialist_source_rl/seed_32401"
for path in \
  "$BASE/config.json" \
  "$P24/alignment_refresh/data/train.sft.jsonl" \
  "$P29/editing_specialist/TRAINING_COMPLETE" \
  "$P29/editing_specialist/adapter/adapter_model.safetensors" \
  "$P251/data/gates/final.jsonl" \
  "$P324/eval/step-000/summary.json" \
  "$P324/eval/step-000/candidates.jsonl"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P34 input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p31_1_frontier_online_rloo:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m py_compile "$SCRIPT_DIR/hard_boundary_reward.py" "$SCRIPT_DIR/train_hard_boundary_rloo.py" "$SCRIPT_DIR/collect_gate.py"
"$PY" -m pytest -q "$SCRIPT_DIR/test_contract.py"
