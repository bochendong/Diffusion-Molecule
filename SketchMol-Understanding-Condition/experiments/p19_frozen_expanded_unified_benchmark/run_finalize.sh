#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="${P19_SCRIPT_DIR:?P19_SCRIPT_DIR must be exported by submitter}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${P19_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUT="${P19_OUTPUT_ROOT:-$PROJECT/outputs/p19_frozen_expanded_unified_benchmark/seed_1919}"
P17_SCRIPT="$PROJECT/experiments/p17_copy_contrastive_unified_benchmark"
P18="$PROJECT/outputs/p18_validity_aware_multinegative_unified/seed_1818_race12"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi
export PYTHONPATH="$P17_SCRIPT:$SCRIPT_DIR:$PROJECT/experiments/unified_smiles_generator${PYTHONPATH:+:$PYTHONPATH}"
test -f "$OUT/P17_GENERATION_COMPLETE"
test -f "$OUT/P18_GENERATION_COMPLETE"
(cd "$OUT/data" && sha256sum -c LOCKED.sha256)
mkdir -p "$OUT/eval"
for model in p17 p18; do
  mkdir -p "$OUT/eval/$model/table1"
  "$PY" "$P17_SCRIPT/annotate_denovo.py" \
    --reference "$OUT/data/denovo_expanded.reference.csv" \
    --raw-candidates "$OUT/generated/$model/denovo.raw.csv" \
    --output "$OUT/eval/$model/denovo.annotated.csv"
  for k in 1 4 8; do
    "$PY" "$PROJECT/scripts/evaluate_moledit_table1_anyk.py" \
      --reference "$OUT/data/table1_expanded.reference.csv" \
      --candidates "$OUT/generated/$model/table1.raw.csv" \
      --output-dir "$OUT/eval/$model/table1/any$k" --candidate-limit "$k" \
      --model-name "P19-$model-expanded-any$k" --task-filter table1 --missing-oracle-policy fail
  done
  "$PY" "$PROJECT/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" \
    --eval-csv "$OUT/data/denovo_expanded.reference.csv" \
    --candidates-csv "$OUT/eval/$model/denovo.annotated.csv" \
    --output-json "$OUT/eval/$model/denovo.metrics.json" \
    --output-md "$OUT/eval/$model/denovo.report.md" --budgets 1,4,8
done
"$PY" "$SCRIPT_DIR/aggregate_expanded.py" \
  --manifest "$OUT/data/manifest.json" --adapter-lock "$OUT/data/ADAPTERS_LOCKED.sha256" \
  --validation-summary "$P18/validation_summary.json" \
  --p18-small-pilot "$P18/PILOT_ESTIMATE.json" \
  --table-reference "$OUT/data/table1_expanded.reference.csv" \
  --p17-table-candidates "$OUT/generated/p17/table1.raw.csv" \
  --p18-table-candidates "$OUT/generated/p18/table1.raw.csv" \
  --p17-denovo-metrics "$OUT/eval/p17/denovo.metrics.json" \
  --p18-denovo-metrics "$OUT/eval/p18/denovo.metrics.json" \
  --p17-denovo-annotated "$OUT/eval/p17/denovo.annotated.csv" \
  --p18-denovo-annotated "$OUT/eval/p18/denovo.annotated.csv" \
  --output "$OUT/EXPANDED_ESTIMATE.json"
touch "$OUT/COMPLETE"
echo "P19 frozen expanded paired estimate complete: $OUT"
