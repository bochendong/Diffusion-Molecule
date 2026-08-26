#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P24_SCRIPT_DIR:?P24_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
P24_OUT="${P24_OUTPUT_ROOT:-$PROJECT/outputs/p24_molprogram_instruct_4m/seed_24003}"
RELEASE="${P24_RELEASE_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule/processed/molprogram-instruct-4m-v1/release}"
BASE="${P24_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
PY="${P24_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P24_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
P17="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
P19="$PROJECT/experiments/p19_frozen_expanded_unified_benchmark"
OUT="$P24_OUT/gate_validation"
test -f "$P24_OUT/gate_13k/TRAINING_COMPLETE"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$P17${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/scratch/bdong/hf_cache/uca_common_llm}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/data" "$OUT/generated"
"$PY" "$SCRIPT_DIR/build_gate_eval.py" --release-root "$RELEASE" \
  --output-jsonl "$OUT/data/prompts.jsonl" --manifest-json "$OUT/data/manifest.json" --rows-per-bucket 10
"$PY" "$P17/generate_pilot.py" --prompts-jsonl "$OUT/data/prompts.jsonl" --base-model "$BASE" \
  --adapter-dir "$P24_OUT/gate_13k/adapter" --output-csv "$OUT/generated/candidates.raw8.csv" --seed 24030
"$PY" "$P19/relabel_generated.py" --csv "$OUT/generated/candidates.raw8.csv" --label p24_gate13k
"$PY" "$SCRIPT_DIR/validate_gate_outputs.py" --training-summary "$P24_OUT/gate_13k/training_summary.json" \
  --prompts-jsonl "$OUT/data/prompts.jsonl" --candidates-csv "$OUT/generated/candidates.raw8.csv" \
  --output-json "$OUT/gate_result.json"
touch "$OUT/GATE_VALIDATION_COMPLETE"
