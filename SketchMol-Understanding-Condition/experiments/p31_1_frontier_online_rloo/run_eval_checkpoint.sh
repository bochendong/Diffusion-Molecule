#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P311_SCRIPT_DIR:?P311_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P311_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P311_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P311_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
P251="$PROJECT/outputs/p25_1_p23_mode_paired_grpo/seed_25125"
OUT="${P311_OUTPUT_ROOT:-$PROJECT/outputs/p31_1_frontier_online_rloo/seed_31101}"
STEP="${P311_STEP:?P311_STEP must be exported}"
TAG="step-$STEP"
BUILD_ADAPTER="$OUT/model/de_novo/checkpoint-$STEP/adapter"
MODIFY_ADAPTER="$OUT/model/edit/checkpoint-$STEP/adapter"
for path in "$OUT/GATE_DATA_COMPLETE" "$BUILD_ADAPTER/adapter_model.safetensors" "$MODIFY_ADAPTER/adapter_model.safetensors"; do
  [[ -f "$path" ]] || { echo "ERROR: missing P31.1 checkpoint gate input: $path" >&2; exit 2; }
done
ORACLE_DIR="${P311_ASSAY_ORACLE_DIR:-$PROJECT/inputs/tdc_oracles}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$PROJECT/experiments/p25_p23_joint_group_rl:$PROJECT/experiments/p23_explicit_task_stage1_v2:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$PROJECT/experiments/p30_1_alignment_raw1_rl/evaluate_small_raw1_gate.py" \
  --prompts-jsonl "$OUT/gate/data/prompts.jsonl" \
  --baseline-summary "$OUT/gate/data/baseline_summary.json" \
  --base-model "$BASE" --adapter-dir "$BUILD_ADAPTER" \
  --output-dir "$OUT/gate/$TAG/denovo" --batch-size 8 \
  --macro-delta-min 0.02 --valid-delta-min -0.01 --bucket-delta-min -0.10
"$PY" "$SCRIPT_DIR/evaluate_edit_gate.py" \
  --gate-jsonl "$P251/data/gates/final.jsonl" \
  --base-model "$BASE" --adapter-dir "$MODIFY_ADAPTER" \
  --output-dir "$OUT/gate/$TAG/edit" --method "p31.1-$TAG" \
  --repeats 4 --batch-size 8 --seed 31151
touch "$OUT/GATE_${STEP}_COMPLETE"
