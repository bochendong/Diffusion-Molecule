#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P26_SCRIPT_DIR:?P26_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P26_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P26_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P26_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P23="${P26_P23_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned}"
P251="${P26_P251_ROOT:-$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125}"
OUT="${P26_OUTPUT_ROOT:-$PROJECT/outputs/p26_decoupled_joint_rl/seed_26001}"
TAG="${P26_EVAL_TAG:?P26_EVAL_TAG must be exported}"
ADAPTER="${P26_EVAL_ADAPTER:?P26_EVAL_ADAPTER must be exported}"
REQUIRED="${P26_EVAL_REQUIRED:?P26_EVAL_REQUIRED must be exported}"
SPLIT="${P26_GATE_SPLIT:-dev}"
case "$SPLIT" in dev|final) ;; *) echo "ERROR: unsupported P26 gate split: $SPLIT" >&2; exit 2 ;; esac
ORACLE_DIR="${P26_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
for path in "$REQUIRED" "$P251/data/gates/$SPLIT.jsonl" "$ADAPTER/adapter_model.safetensors" \
  "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P26 eval input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$PROJECT/experiments/p25_p23_joint_group_rl/evaluate_frozen_gate.py" \
  --gate-jsonl "$P251/data/gates/$SPLIT.jsonl" --base-model "$BASE" \
  --adapter-dir "$ADAPTER" --output-dir "$OUT/gate/$SPLIT/$TAG" \
  --method "$TAG" --repeats 4 --batch-size 8 --seed 251251
touch "$OUT/GATE_${SPLIT}_${TAG}_COMPLETE"
