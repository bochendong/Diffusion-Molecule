#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P321_SCRIPT_DIR:?P321_SCRIPT_DIR must be exported}"
STEP="${P321_STEP:?P321_STEP must be exported as 000, 010, 020, or 030}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P321_OUTPUT_ROOT:-$PROJECT/outputs/p32_1_verifier_routed_residual_rl/seed_32101}"
PY="${P321_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P321_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
ORACLE_DIR="$PROJECT/inputs/tdc_oracles"
P32="$PROJECT/outputs/p32_unified_graph_repair_rl/seed_32001"
if [[ "$STEP" == "000" ]]; then
  ADAPTER="$P32/model/checkpoint-030/adapter"
else
  ADAPTER="$OUT/model/checkpoint-$STEP/adapter"
fi
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p32_unified_graph_repair_rl:$PROJECT/experiments/unified_constraint_agent:$PROJECT/experiments/p31_1_frontier_online_rloo:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
"$PY" "$SCRIPT_DIR/evaluate_residual_checkpoint.py" \
  --gate-jsonl "$OUT/data/gate.jsonl" \
  --base-model Qwen/Qwen2.5-1.5B-Instruct --adapter-dir "$ADAPTER" \
  --output-dir "$OUT/eval/step-$STEP" --tag "$STEP" \
  --max-steps 2 --max-actions 16 --site-limit 24 \
  --score-batch-size 4 --max-length 768
