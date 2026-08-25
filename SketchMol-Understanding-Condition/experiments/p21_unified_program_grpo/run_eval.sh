#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P21_SCRIPT_DIR:?P21_SCRIPT_DIR must be exported by submitter}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P21_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
DEP="${P21_DEP_OVERLAY:-/scratch/bdong/venvs/uca_common_llm_overlay}"
BASE="${P21_BASE_MODEL:-/scratch/bdong/checkpoints/Qwen2.5-VL-7B-Instruct}"
OUT="${P21_OUTPUT_ROOT:-$PROJECT/outputs/p21_unified_program_grpo/seed_2121}"
P17="$PROJECT/outputs/p17_copy_contrastive_unified_benchmark/seed_1717"
P19="$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919"
P17_SCRIPT="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
P19_SCRIPT="$PROJECT/experiments/p19_frozen_expanded_unified_benchmark"
ADAPTER="$OUT/model/p21/adapter"
test -f "$OUT/TRAIN_COMPLETE"
sha256sum -c "$OUT/model/p21/FROZEN.sha256"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6; fi
export PYTHONPATH="$DEP:$P17_SCRIPT:$P19_SCRIPT:$SCRIPT_DIR:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/bdong/hf_cache/uca_common_llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/eval" "$OUT/generated"
"$PY" "$P17_SCRIPT/evaluate_expanded.py" --dev-jsonl "$P17/data/dev.id_condition_source_isolated.jsonl" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" --label p21_grpo --view id --output-json "$OUT/eval/p21.id.json" --fixed-k 3 --seed 1717
"$PY" "$P17_SCRIPT/evaluate_expanded.py" --dev-jsonl "$P17/data/dev.condition_source_ood.jsonl" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" --label p21_grpo --view ood --output-json "$OUT/eval/p21.ood.json" --fixed-k 3 --seed 1717
"$PY" "$P17_SCRIPT/generate_pilot.py" --prompts-jsonl "$P19/data/table1_expanded.prompts.jsonl" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" --output-csv "$OUT/generated/table1.raw.csv" --seed 2717
"$PY" "$P17_SCRIPT/generate_pilot.py" --prompts-jsonl "$P19/data/denovo_expanded.prompts.jsonl" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" --output-csv "$OUT/generated/denovo.raw.csv" --seed 3717
"$PY" "$P17_SCRIPT/annotate_denovo.py" --reference "$P19/data/denovo_expanded.reference.csv" \
  --raw-candidates "$OUT/generated/denovo.raw.csv" --output "$OUT/eval/denovo.annotated.csv"
"$PY" "$PROJECT/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
  --eval-csv "$P19/data/denovo_expanded.reference.csv" --candidates-csv "$OUT/eval/denovo.annotated.csv" \
  --output-json "$OUT/eval/denovo.metrics.json" --output-md "$OUT/eval/denovo.report.md" --budgets 1,4,8
"$PY" "$SCRIPT_DIR/aggregate_p21.py" \
  --table-reference "$P19/data/table1_expanded.reference.csv" \
  --p18-table "$P19/generated/p18/table1.raw.csv" --p21-table "$OUT/generated/table1.raw.csv" \
  --p18-denovo-metrics "$P19/eval/p18/denovo.metrics.json" --p21-denovo-metrics "$OUT/eval/denovo.metrics.json" \
  --p18-denovo-annotated "$P19/eval/p18/denovo.annotated.csv" --p21-denovo-annotated "$OUT/eval/denovo.annotated.csv" \
  --p21-id "$OUT/eval/p21.id.json" --p21-ood "$OUT/eval/p21.ood.json" \
  --training-summary "$OUT/model/p21/training_summary.json" --output "$OUT/P21_ESTIMATE.json"
touch "$OUT/COMPLETE"
echo "P21 frozen estimate complete"
