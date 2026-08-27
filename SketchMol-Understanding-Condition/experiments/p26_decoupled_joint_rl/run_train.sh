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
TRAIN_JSONL="${P26_TRAIN_JSONL:-$P23/data/train.sft.jsonl}"
INPUT_ADAPTER="${P26_INPUT_ADAPTER:-$P23/model/stage1_v2/adapter}"
POLICY_OUTPUT_DIR="${P26_POLICY_OUTPUT_DIR:-$OUT/model/p26}"
TRAIN_SEED="${P26_TRAIN_SEED:-26001}"
ORACLE_DIR="${P26_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
for path in "$P251/PREPARED" "$TRAIN_JSONL" \
  "$INPUT_ADAPTER/adapter_model.safetensors" \
  "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P26 training input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/train_decoupled_joint_rl.py" \
  --train-jsonl "$TRAIN_JSONL" --base-model "$BASE" \
  --input-adapter "$INPUT_ADAPTER" --output-dir "$POLICY_OUTPUT_DIR" \
  --pairs 30 --group-size 16 --learning-rate 5e-7 \
  --denovo-anchor-weight 1.0 --edit-anchor-weight 1.0 \
  --reference-kl-weight 0.05 --gradient-surgery pcgrad \
  --checkpoint-every 10 --seed "$TRAIN_SEED"
sha256sum "$POLICY_OUTPUT_DIR/adapter/adapter_model.safetensors" > "$POLICY_OUTPUT_DIR/FROZEN.sha256"
touch "$OUT/TRAIN_COMPLETE"
