#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P33_SCRIPT_DIR:?P33_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P33_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P33_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P33_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
P251="$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125"
for path in "$BASE/config.json" "$P24/alignment_refresh/data/train.sft.jsonl" "$P251/data/gates/final.jsonl"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P33 input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m py_compile "$SCRIPT_DIR/prepare_pilot.py" "$SCRIPT_DIR/train_arm.py" "$SCRIPT_DIR/evaluate_arm.py" "$SCRIPT_DIR/collect_pilot.py"
"$PY" -m pytest -q "$SCRIPT_DIR/test_contract.py"
