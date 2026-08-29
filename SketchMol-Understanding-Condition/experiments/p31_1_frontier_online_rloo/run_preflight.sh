#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P311_SCRIPT_DIR:?P311_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P311_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P311_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
P251="$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125"
P303="$PROJECT/outputs/p30_3_joint_multinegative_refinement/seed_30301"
for path in \
  "$P24/alignment_refresh/ALIGNMENT_REFRESH_COMPLETE" \
  "$P24/alignment_refresh/model/adapter/adapter_model.safetensors" \
  "$P24/alignment_refresh/data/train.sft.jsonl" \
  "$P251/data/gates/final.jsonl" \
  "$P303/edit_eval/gate/final/baseline/summary.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P31.1 input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m py_compile \
  "$SCRIPT_DIR/rloo_math.py" \
  "$SCRIPT_DIR/train_online_rloo.py" \
  "$SCRIPT_DIR/evaluate_edit_gate.py" \
  "$SCRIPT_DIR/collect_joint_gate.py"
"$PY" -m pytest -q "$SCRIPT_DIR/test_contract.py"
