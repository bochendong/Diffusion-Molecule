#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P302_SCRIPT_DIR:?P302_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P302_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P302_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P302_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P23="$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned"
P24="$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003"
INPUT_ADAPTER="${P302_INPUT_ADAPTER:-$P24/alignment_refresh/model/adapter}"
P24_MARKER="${P302_P24_MARKER:-$P24/alignment_refresh/ALIGNMENT_REFRESH_COMPLETE}"
OUT="${P302_OUTPUT_ROOT:-$PROJECT/outputs/p30_2_validity_stable_raw1_rl/seed_30201}"
TRAIN_JSONL="${P302_TRAIN_JSONL:-$P23/data/train.sft.jsonl}"
MODEL_DIR="$OUT/model/validity_stable_rl"
ORACLE_DIR="${P302_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
for path in "$P24_MARKER" "$TRAIN_JSONL" "$INPUT_ADAPTER/adapter_model.safetensors" "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P30.2 input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p30_balanced_shared_policy_rl:$PROJECT/experiments/p26_decoupled_joint_rl:$PROJECT/experiments/p25_1_p23_mode_paired_grpo:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/train_validity_stable_rl.py" \
  --train-jsonl "$TRAIN_JSONL" --base-model "$BASE" \
  --input-adapter "$INPUT_ADAPTER" --output-dir "$MODEL_DIR" \
  --pairs 30 --group-size 16 --learning-rate 1.0e-7 \
  --denovo-anchor-weight 2.0 --edit-anchor-weight 1.5 \
  --reference-kl-weight 0.20 --grad-clip 0.5 \
  --checkpoint-every 10 --seed 30201
for step in 010 020 030; do
  test -f "$MODEL_DIR/checkpoint-$step/CHECKPOINT_COMPLETE"
done
sha256sum "$MODEL_DIR/adapter/adapter_model.safetensors" > "$MODEL_DIR/FROZEN_STEP30.sha256"
touch "$OUT/TRAIN_COMPLETE"

