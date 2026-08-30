#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P34_SCRIPT_DIR:?P34_SCRIPT_DIR must be exported}"
STEP="${P34_STEP:?P34_STEP must be 005, 010, or 020}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${P34_OUTPUT_ROOT:-$PROJECT/outputs/p34_hard_boundary_edit_rl/seed_34001}"
PY="${P34_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P34_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P34_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P251="$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125"
ORACLE_DIR="${P34_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
ADAPTER="$OUT/model/edit/checkpoint-$STEP/adapter"
test -f "$ADAPTER/adapter_model.safetensors"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$PROJECT/experiments/p31_1_frontier_online_rloo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
"$PY" "$PROJECT/experiments/p31_1_frontier_online_rloo/evaluate_edit_gate.py" \
  --gate-jsonl "$P251/data/gates/final.jsonl" --base-model "$BASE" \
  --adapter-dir "$ADAPTER" --output-dir "$OUT/eval/step-$STEP" \
  --method "p34-step-$STEP" --repeats 1 --batch-size 8 --seed 23501
