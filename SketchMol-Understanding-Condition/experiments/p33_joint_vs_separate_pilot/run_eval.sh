#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P33_SCRIPT_DIR:?P33_SCRIPT_DIR must be exported}"
ARM="${P33_ARM:?P33_ARM must be joint, denovo, or edit}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P33_OUTPUT_ROOT:-$PROJECT/outputs/p33_joint_vs_separate_pilot/seed_33001}"
PY="${P33_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P33_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P33_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P251="$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125"
ORACLE_DIR="${P33_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
"$PY" "$SCRIPT_DIR/evaluate_arm.py" --denovo-gate "$OUT/data/gate.denovo.jsonl" \
  --edit-gate "$P251/data/gates/final.jsonl" --base-model "$BASE" \
  --adapter-dir "$OUT/model/$ARM/adapter" --output-dir "$OUT/eval/$ARM" \
  --arm "$ARM" --batch-size 8 --seed 33051
