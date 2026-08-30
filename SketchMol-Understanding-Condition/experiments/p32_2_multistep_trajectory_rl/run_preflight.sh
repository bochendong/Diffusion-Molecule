#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P322_SCRIPT_DIR:?P322_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P322_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P322_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
P32="$PROJECT/outputs/p32_unified_graph_repair_rl/seed_32001"
P321="$PROJECT/outputs/p32_1_verifier_routed_residual_rl/seed_32101"
ORACLE_DIR="$PROJECT/inputs/tdc_oracles"
for path in \
  "$P32/model/checkpoint-030/adapter/adapter_model.safetensors" \
  "$P321/data/train.jsonl" \
  "$P321/data/gate.jsonl" \
  "$P321/data/manifest.json" \
  "$P321/support_audit/result.json" \
  "$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl" \
  "$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P32.2 input: $path" >&2; exit 2; }
done
grep -q '"decision": "RUN_RESIDUAL_RL"' "$P321/support_audit/result.json"
grep -q '"direct_labels_recomputed_with_pinned_oracles": true' "$P321/data/manifest.json"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p32_1_verifier_routed_residual_rl:$PROJECT/experiments/p32_unified_graph_repair_rl:$PROJECT/experiments/unified_constraint_agent:$PROJECT/experiments/p31_1_frontier_online_rloo:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export SUCC_GSK3B_ORACLE_PATH="$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl"
export SUCC_DRD2_ORACLE_PATH="$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl"
"$PY" -m py_compile "$SCRIPT_DIR"/*.py
"$PY" -m pytest -q "$SCRIPT_DIR/test_contract.py"
