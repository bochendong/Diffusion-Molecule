#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P28_SCRIPT_DIR:?P28_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P28_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P28_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P28_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
INPUT_ADAPTER="${P28_INPUT_ADAPTER:?P28_INPUT_ADAPTER must be exported}"
OUT="${P28_OUTPUT_ROOT:?P28_OUTPUT_ROOT must be exported}"
VARIANT="${P28_VARIANT:?P28_VARIANT must be exported}"
TRAIN_JSONL="${P28_TRAIN_JSONL:-$P23/data/train.sft.jsonl}"
ORACLE_DIR="${P28_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
POLICY_DIR="$OUT/model/$VARIANT"

for path in "$TRAIN_JSONL" "$INPUT_ADAPTER/adapter_model.safetensors" \
  "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P28 input: $path" >&2; exit 2; }
done

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

case "$VARIANT" in
  vanilla_paired_grpo)
    "$PY" "$PROJECT/experiments/p25_1_p23_mode_paired_grpo/train_mode_paired_grpo.py" \
      --train-jsonl "$TRAIN_JSONL" --base-model "$BASE" \
      --input-adapter "$INPUT_ADAPTER" --output-dir "$POLICY_DIR" \
      --pairs 30 --group-size 16 --learning-rate 5e-7 \
      --sft-anchor-weight 1.0 --reference-kl-weight 0.05 \
      --checkpoint-every 10 --seed 28001
    ;;
  decoupled_no_pcgrad)
    "$PY" "$PROJECT/experiments/p26_decoupled_joint_rl/train_decoupled_joint_rl.py" \
      --train-jsonl "$TRAIN_JSONL" --base-model "$BASE" \
      --input-adapter "$INPUT_ADAPTER" --output-dir "$POLICY_DIR" \
      --pairs 30 --group-size 16 --learning-rate 5e-7 \
      --denovo-anchor-weight 1.0 --edit-anchor-weight 1.0 \
      --reference-kl-weight 0.05 --gradient-surgery none \
      --checkpoint-every 10 --seed 28001
    ;;
  editing_protected_pcgrad)
    "$PY" "$PROJECT/experiments/p26_decoupled_joint_rl/train_decoupled_joint_rl.py" \
      --train-jsonl "$TRAIN_JSONL" --base-model "$BASE" \
      --input-adapter "$INPUT_ADAPTER" --output-dir "$POLICY_DIR" \
      --pairs 30 --group-size 16 --learning-rate 2.5e-7 \
      --denovo-anchor-weight 1.0 --edit-anchor-weight 2.0 \
      --reference-kl-weight 0.10 --gradient-surgery pcgrad \
      --checkpoint-every 10 --seed 28001
    ;;
  *)
    echo "ERROR: unsupported P28 variant: $VARIANT" >&2
    exit 2
    ;;
esac

sha256sum "$POLICY_DIR/checkpoint-020/adapter/adapter_model.safetensors" \
  > "$POLICY_DIR/FROZEN_STEP20.sha256"
sha256sum "$POLICY_DIR/adapter/adapter_model.safetensors" \
  > "$POLICY_DIR/FROZEN_STEP30.sha256"
touch "$OUT/TRAIN_${VARIANT}_COMPLETE"
