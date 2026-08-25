#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P17_SCRIPT_DIR:?P17_SCRIPT_DIR must be exported by submitter}"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"; PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"
PY="${P17_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P17_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P17_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P17_OUTPUT_ROOT:-$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717}"
ADAPTER="$OUT/model/p17/adapter"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6; fi
export PYTHONPATH="$DEP:$SCRIPT_DIR:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}" HF_HOME=/scratch/bdong/hf_cache/uca_common_llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for view in id ood; do
  file="$OUT/data/dev.$([[ "$view" == id ]] && echo id_condition_source_isolated || echo condition_source_ood).jsonl"
  "$PY" "$SCRIPT_DIR/evaluate_expanded.py" --dev-jsonl "$file" --base-model "$BASE" --adapter-dir "$ADAPTER" \
    --label p17_optimized --view "$view" --output-json "$OUT/eval/p17.$view.json" --fixed-k 3 --seed 1717
done
"$PY" "$SCRIPT_DIR/aggregate_validation.py" --p16-id "$OUT/eval/p16.id.json" --p17-id "$OUT/eval/p17.id.json" \
  --p16-ood "$OUT/eval/p16.ood.json" --p17-ood "$OUT/eval/p17.ood.json" --manifest "$OUT/data/manifest.json" --output "$OUT/validation_summary.json"

mkdir -p "$OUT/pilot/generated" "$OUT/pilot/table1"
"$PY" "$SCRIPT_DIR/generate_pilot.py" --prompts-jsonl "$OUT/pilot/table1_pilot.prompts.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" --output-csv "$OUT/pilot/generated/table1.raw.csv" --seed 2717
"$PY" "$SCRIPT_DIR/generate_pilot.py" --prompts-jsonl "$OUT/pilot/denovo_pilot.prompts.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" --output-csv "$OUT/pilot/generated/denovo.raw.csv" --seed 3717
"$PY" "$SCRIPT_DIR/annotate_denovo.py" --reference "$OUT/pilot/denovo_pilot.reference.csv" --raw-candidates "$OUT/pilot/generated/denovo.raw.csv" --output "$OUT/pilot/generated/denovo.annotated.csv"
for k in 1 4 8; do
  "$PY" "$PROJECT/scripts/evaluate_moledit_table1_anyk.py" --reference "$OUT/pilot/table1_pilot.reference.csv" \
    --candidates "$OUT/pilot/generated/table1.raw.csv" --output-dir "$OUT/pilot/table1/any$k" --candidate-limit "$k" \
    --model-name "P17-pilot-any$k" --task-filter table1 --missing-oracle-policy fail
done
"$PY" "$PROJECT/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
  --eval-csv "$OUT/pilot/denovo_pilot.reference.csv" --candidates-csv "$OUT/pilot/generated/denovo.annotated.csv" \
  --output-json "$OUT/pilot/denovo.metrics.json" --output-md "$OUT/pilot/denovo.report.md" --budgets 1,4,8
"$PY" "$SCRIPT_DIR/aggregate_pilot.py" --gate "$OUT/validation_summary.json" --pilot-manifest "$OUT/pilot/manifest.json" \
  --table1-root "$OUT/pilot/table1" --denovo "$OUT/pilot/denovo.metrics.json" --output "$OUT/PILOT_ESTIMATE.json"
touch "$OUT/COMPLETE"
echo "P17 expanded validation and pilot estimate complete: $OUT"
