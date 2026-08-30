#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P34_SCRIPT_DIR:?P34_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P34_OUTPUT_ROOT:-$PROJECT/outputs/p34_hard_boundary_edit_rl/seed_34001}"
PY="${P34_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P34_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P34_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
P29="$PROJECT/outputs/p29_unified_vs_specialist_ablation/seed_24003"
ORACLE_DIR="${P34_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p31_1_frontier_online_rloo:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
"$PY" "$SCRIPT_DIR/train_hard_boundary_rloo.py" \
  --train-jsonl "$P24/alignment_refresh/data/train.sft.jsonl" \
  --base-model "$BASE" --input-adapter "$P29/editing_specialist/adapter" \
  --output-dir "$OUT/model/edit" --target-updates 20 --max-attempts 600 \
  --group-size 16 --learning-rate 1e-7 --reference-kl-weight 0.1 \
  --grad-clip 0.5 --checkpoint-updates 5,10,20 --seed 34001
for step in 005 010 020; do
  test -f "$OUT/model/edit/checkpoint-$step/CHECKPOINT_COMPLETE"
done
