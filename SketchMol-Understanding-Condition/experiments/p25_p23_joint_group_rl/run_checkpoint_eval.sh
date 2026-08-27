#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P25_SCRIPT_DIR:?P25_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P25_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P25_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P25_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P25_OUTPUT_ROOT:-$PROJECT/outputs/p25_p23_joint_group_rl/seed_2525}"
INDEX="${P25_CHECKPOINT_INDEX:?P25_CHECKPOINT_INDEX must be 013 or 026}"
case "$INDEX" in 013|026) ;; *) echo "ERROR: unsupported checkpoint $INDEX" >&2; exit 2 ;; esac
ADAPTER="$OUT/model/p25/checkpoint-$INDEX/adapter"
ORACLE_DIR="${P25_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
for path in "$OUT/GATE_COMPLETE" "$OUT/data/frozen_gate.jsonl" \
  "$ADAPTER/adapter_model.safetensors" "$SUCC_GSK3B_ORACLE_PATH" "$SUCC_DRD2_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P25 diagnostic input: $path" >&2; exit 2; }
done
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/evaluate_frozen_gate.py" \
  --gate-jsonl "$OUT/data/frozen_gate.jsonl" --base-model "$BASE" \
  --adapter-dir "$ADAPTER" --output-dir "$OUT/gate/checkpoint-$INDEX" \
  --method "checkpoint-$INDEX" --repeats 4 --batch-size 8 --seed 25251
touch "$OUT/GATE_CHECKPOINT_${INDEX}_COMPLETE"
