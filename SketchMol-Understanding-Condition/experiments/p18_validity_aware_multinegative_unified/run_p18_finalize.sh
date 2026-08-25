#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P18_SCRIPT_DIR:?P18_SCRIPT_DIR must be exported by submitter}"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"; PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"
PY="${P18_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P18_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P18_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P18_OUTPUT_ROOT:-$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818}"
P17="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717"
P17_SCRIPT="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
ADAPTER="$OUT/model/p18/adapter"
test -f "$OUT/P18_TRAIN_COMPLETE"; sha256sum -c "$OUT/model/p18/FROZEN.sha256"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6; fi
export PYTHONPATH="$DEP:$P17_SCRIPT:$SCRIPT_DIR:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}" HF_HOME=/scratch/bdong/hf_cache/uca_common_llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/eval" "$OUT/pilot/generated" "$OUT/pilot/table1"
for view in id ood; do
  file="$P17/data/dev.$([[ "$view" == id ]] && echo id_condition_source_isolated || echo condition_source_ood).jsonl"
  "$PY" "$P17_SCRIPT/evaluate_expanded.py" --dev-jsonl "$file" --base-model "$BASE" --adapter-dir "$ADAPTER" \
    --label p18_multinegative --view "$view" --output-json "$OUT/eval/p18.$view.json" --fixed-k 3 --seed 1717
done
"$PY" "$SCRIPT_DIR/aggregate_validation.py" \
  --p16-id "$P17/eval/p16.id.json" --p16-ood "$P17/eval/p16.ood.json" \
  --p17-id "$P17/eval/p17.id.json" --p17-ood "$P17/eval/p17.ood.json" \
  --p18-id "$OUT/eval/p18.id.json" --p18-ood "$OUT/eval/p18.ood.json" \
  --p17-manifest "$P17/data/manifest.json" --input-audit "$OUT/data/locked_input_audit.json" --output "$OUT/validation_summary.json"

"$PY" "$P17_SCRIPT/generate_pilot.py" --prompts-jsonl "$P17/pilot/table1_pilot.prompts.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" --output-csv "$OUT/pilot/generated/table1.raw.csv" --seed 2717
"$PY" "$SCRIPT_DIR/relabel_generated.py" --csv "$OUT/pilot/generated/table1.raw.csv"
"$PY" "$P17_SCRIPT/generate_pilot.py" --prompts-jsonl "$P17/pilot/denovo_pilot.prompts.jsonl" --base-model "$BASE" --adapter-dir "$ADAPTER" --output-csv "$OUT/pilot/generated/denovo.raw.csv" --seed 3717
"$PY" "$SCRIPT_DIR/relabel_generated.py" --csv "$OUT/pilot/generated/denovo.raw.csv"
"$PY" "$P17_SCRIPT/annotate_denovo.py" --reference "$P17/pilot/denovo_pilot.reference.csv" --raw-candidates "$OUT/pilot/generated/denovo.raw.csv" --output "$OUT/pilot/generated/denovo.annotated.csv"
for k in 1 4 8; do
  "$PY" "$PROJECT/scripts/evaluate_moledit_table1_anyk.py" --reference "$P17/pilot/table1_pilot.reference.csv" \
    --candidates "$OUT/pilot/generated/table1.raw.csv" --output-dir "$OUT/pilot/table1/any$k" --candidate-limit "$k" \
    --model-name "P18-pilot-any$k" --task-filter table1 --missing-oracle-policy fail
done
"$PY" "$PROJECT/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
  --eval-csv "$P17/pilot/denovo_pilot.reference.csv" --candidates-csv "$OUT/pilot/generated/denovo.annotated.csv" \
  --output-json "$OUT/pilot/denovo.metrics.json" --output-md "$OUT/pilot/denovo.report.md" --budgets 1,4,8
"$PY" "$SCRIPT_DIR/aggregate_pilot.py" --gate "$OUT/validation_summary.json" \
  --p17-table-root "$P17/pilot/table1" --p18-table-root "$OUT/pilot/table1" \
  --p17-denovo-metrics "$P17/pilot/denovo.metrics.json" --p18-denovo-metrics "$OUT/pilot/denovo.metrics.json" \
  --p17-denovo-annotated "$P17/pilot/generated/denovo.annotated.csv" --p18-denovo-annotated "$OUT/pilot/generated/denovo.annotated.csv" \
  --output "$OUT/PILOT_ESTIMATE.json"
touch "$OUT/COMPLETE"
echo "P18 paired validation and pilot complete: $OUT"
