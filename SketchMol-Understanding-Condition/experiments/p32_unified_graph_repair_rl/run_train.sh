#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P32_SCRIPT_DIR:?P32_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P32_OUTPUT_ROOT:-$PROJECT/outputs/p32_unified_graph_repair_rl/seed_32001}"
PY="${P32_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P32_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
ORACLE_DIR="$PROJECT/inputs/tdc_oracles"
COMMON="$PROJECT/outputs/unified_constraint_agent_common_llm_pilot_v1/model/seed_1703/adapter"
[[ -f "$OUT/SUPPORT_AUDIT_PASS" ]] || { echo "ERROR: P32 support audit did not pass" >&2; exit 3; }
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/unified_constraint_agent:$PROJECT/experiments/p31_1_frontier_online_rloo:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_legacy_sklearn_compatible.pkl}"
"$PY" "$SCRIPT_DIR/train_shared_graph_repair_rl.py" \
  --train-jsonl "$OUT/data/train.jsonl" \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --input-adapter "$COMMON" \
  --output-dir "$OUT/model" \
  --pairs 30 --max-steps 2 --max-actions 16 --site-limit 24 \
  --score-batch-size 4 --max-length 768 --temperature 0.8 \
  --learning-rate 3e-6 --grad-clip 1.0 --checkpoint-steps 10,20,30 \
  --seed 32001
