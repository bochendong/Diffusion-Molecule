#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P251_SCRIPT_DIR:?P251_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P251_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P251_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P251_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P23="${P251_P23_ROOT:-$PROJECT/outputs/p23_explicit_task_stage1_v2/seed_2323_full24k_aligned}"
OUT="${P251_OUTPUT_ROOT:-$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125}"
ORACLE_DIR="${P251_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
for path in "$OUT/PREPARED" "$P23/data/train.sft.jsonl" \
  "$P23/model/stage1_v2/adapter/adapter_model.safetensors" \
  "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P25.1 training input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/train_mode_paired_grpo.py" \
  --train-jsonl "$P23/data/train.sft.jsonl" --base-model "$BASE" \
  --input-adapter "$P23/model/stage1_v2/adapter" --output-dir "$OUT/model/p25_1" \
  --pairs 30 --group-size 8 --learning-rate 5e-7 --sft-anchor-weight 1.0 \
  --reference-kl-weight 0.05 --checkpoint-every 10 --seed 25125
sha256sum "$OUT/model/p25_1/adapter/adapter_model.safetensors" > "$OUT/model/p25_1/FROZEN.sha256"
touch "$OUT/TRAIN_COMPLETE"
