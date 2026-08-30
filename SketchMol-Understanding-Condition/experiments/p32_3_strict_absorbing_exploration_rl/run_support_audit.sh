#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P323_SCRIPT_DIR:?P323_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P323_OUTPUT_ROOT:-$PROJECT/outputs/p32_3_strict_absorbing_exploration_rl/seed_32301}"
PY="${P323_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P323_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
P321="$PROJECT/outputs/p32_1_verifier_routed_residual_rl/seed_32101"
ORACLE_DIR="$PROJECT/inputs/tdc_oracles"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p32_1_verifier_routed_residual_rl:$PROJECT/experiments/p32_unified_graph_repair_rl:$PROJECT/experiments/unified_constraint_agent:$PROJECT/experiments/p31_1_frontier_online_rloo:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
"$PY" "$PROJECT/experiments/p32_1_verifier_routed_residual_rl/audit_residual_support.py" \
  --gate-jsonl "$P321/data/train.jsonl" --output-dir "$OUT/support_audit" \
  --steps 2 --max-actions 16 --site-limit 24 --beam-size 16 \
  --minimum-rescues 5 --require-pass
touch "$OUT/SUPPORT_AUDIT_PASS"
