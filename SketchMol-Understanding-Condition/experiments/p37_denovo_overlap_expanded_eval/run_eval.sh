#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P37_SCRIPT_DIR:?P37_SCRIPT_DIR must be exported}"
SCALE="${P37_SCALE:?P37_SCALE must be 10000 or 100000}"
ARM="${P37_ARM:?P37_ARM must be joint or denovo}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P37_OUTPUT_ROOT:-$PROJECT/outputs/p37_denovo_overlap_expanded_eval/seed_37101}"
PY="${P37_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P37_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P37_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
ORACLE_DIR="${P37_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
case "$SCALE" in
  10000) MODEL_ROOT="$PROJECT/outputs/p33_joint_vs_separate_10k_single_seed/seed_33101/model" ;;
  100000) MODEL_ROOT="$PROJECT/outputs/p35_joint_vs_separate_scale_sweep/scale_100000/seed_33101/model" ;;
  *) echo "unsupported P37 scale: $SCALE" >&2; exit 2 ;;
esac
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p33_joint_vs_separate_pilot:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
"$PY" "$SCRIPT_DIR/evaluate_overlap_arm.py" \
  --gate "$OUT/data/gate.denovo_overlap.jsonl" --base-model "$BASE" \
  --adapter-dir "$MODEL_ROOT/$ARM/adapter" \
  --output-dir "$OUT/scale_$SCALE/eval/$ARM" --arm "$ARM" --scale "$SCALE" \
  --batch-size 8 --seed 37151
