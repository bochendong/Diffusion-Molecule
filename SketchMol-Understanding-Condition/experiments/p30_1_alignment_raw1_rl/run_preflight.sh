#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P301_SCRIPT_DIR:?P301_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P301_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P301_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m py_compile "$SCRIPT_DIR/build_small_raw1_gate.py" "$SCRIPT_DIR/evaluate_small_raw1_gate.py"
"$PY" -m pytest -q "$SCRIPT_DIR/test_contract.py"

