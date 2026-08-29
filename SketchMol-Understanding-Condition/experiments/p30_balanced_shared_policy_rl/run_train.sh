#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P30_SCRIPT_DIR:?P30_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P30_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P30_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P30_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P23="${P30_P23_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned}"
INPUT_ADAPTER="${P30_INPUT_ADAPTER:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003/full/adapter}"
P24_MARKER="${P30_P24_MARKER:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003/full/TRAINING_COMPLETE}"
OUT="${P30_OUTPUT_ROOT:-$PROJECT/outputs/p30_balanced_shared_policy_rl/seed_30001}"
TRAIN_JSONL="${P30_TRAIN_JSONL:-$P23/data/train.sft.jsonl}"
MODEL_DIR="$OUT/model/balanced_shared_rl"
ORACLE_DIR="${P30_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
for path in "$P24_MARKER" "$TRAIN_JSONL" "$INPUT_ADAPTER/adapter_model.safetensors" \
  "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P30 input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/train_balanced_shared_rl.py" \
  --train-jsonl "$TRAIN_JSONL" --base-model "$BASE" \
  --input-adapter "$INPUT_ADAPTER" --output-dir "$MODEL_DIR" \
  --pairs 60 --group-size 16 --learning-rate 1.5e-7 \
  --denovo-anchor-weight 1.5 --edit-anchor-weight 1.5 \
  --reference-kl-weight 0.10 --grad-clip 0.5 \
  --checkpoint-every 10 --seed 30001
for step in 010 020 030 040 050 060; do
  test -f "$MODEL_DIR/checkpoint-$step/CHECKPOINT_COMPLETE"
  test -f "$MODEL_DIR/checkpoint-$step/adapter/adapter_model.safetensors"
done
sha256sum "$MODEL_DIR/adapter/adapter_model.safetensors" > "$MODEL_DIR/FROZEN_STEP60.sha256"
touch "$OUT/TRAIN_COMPLETE"
