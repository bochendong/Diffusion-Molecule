#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P311_SCRIPT_DIR:?P311_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P311_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P311_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P311_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
OUT="${P311_OUTPUT_ROOT:-$PROJECT/outputs/p31_1_frontier_online_rloo/seed_31101}"
MODE="${P311_MODE:?P311_MODE must be exported}"
case "$MODE" in de_novo|edit) ;; *) echo "ERROR: unsupported P311_MODE=$MODE" >&2; exit 2 ;; esac
ORACLE_DIR="${P311_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/train_online_rloo.py" \
  --mode "$MODE" \
  --train-jsonl "$P24/alignment_refresh/data/train.sft.jsonl" \
  --base-model "$BASE" \
  --input-adapter "$P24/alignment_refresh/model/adapter" \
  --output-dir "$OUT/model/$MODE" \
  --target-updates 100 --max-attempts 1200 --group-size 16 \
  --learning-rate 2e-7 --reference-kl-weight 0.05 --grad-clip 0.5 \
  --checkpoint-updates 25,50,100 --seed 31101
for step in 025 050 100; do
  test -f "$OUT/model/$MODE/checkpoint-$step/CHECKPOINT_COMPLETE"
done
touch "$OUT/TRAIN_${MODE}_COMPLETE"
