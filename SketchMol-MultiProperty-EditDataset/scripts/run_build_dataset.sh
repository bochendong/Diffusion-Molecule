#!/usr/bin/env bash
# Build the large multi-property edit dataset on a compute node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # Slurm batch shells do not always initialize Environment Modules.
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
else
  echo "WARNING: Environment Modules are unavailable; relying on PYTHONPATH/venv for RDKit." >&2
fi

PYTHON_BIN="${SMMED_PYTHON_BIN:-${PYTHON_BIN:-python}}"
export PYTHONPATH="$PROJECT_DIR:$REPO_DIR/SketchMol-Understanding-Condition${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from rdkit import Chem
PY
then
  echo "ERROR: RDKit is not importable with PYTHON_BIN=$PYTHON_BIN" >&2
  echo "       Use the module Python by leaving SMMED_PYTHON_BIN unset, or point it at a Python with RDKit." >&2
  exit 2
fi

INPUT_CSV="${SMMED_INPUT_CSV:-PhysTabMol/runs/20260601_070814_sketchmol_compare_structure_seed7/tables/train_table.csv}"
OUTPUT_DIR="${SMMED_OUTPUT_DIR:-SketchMol-MultiProperty-EditDataset/outputs/multiproperty_100k_v1}"
LIMIT="${SMMED_LIMIT:-100000}"
MAX_PAIRS="${SMMED_MAX_PAIRS:-100000}"
MAX_PAIRS_PER_SCAFFOLD="${SMMED_MAX_PAIRS_PER_SCAFFOLD:-300}"
MAX_MOLS_PER_SCAFFOLD="${SMMED_MAX_MOLECULES_PER_SCAFFOLD:-300}"
MIN_ACTIVE_PROPERTIES="${SMMED_MIN_ACTIVE_PROPERTIES:-2}"
THRESHOLD_SCALE="${SMMED_THRESHOLD_SCALE:-1.0}"
MIN_SIMILARITY="${SMMED_MIN_SIMILARITY:-0.2}"
MAX_SIMILARITY="${SMMED_MAX_SIMILARITY:-0.9}"
EVAL_FRACTION="${SMMED_EVAL_FRACTION:-0.2}"
CONDITIONS_PER_PAIR="${SMMED_CONDITIONS_PER_PAIR:-3}"
MIN_CONDITION_PROPERTIES="${SMMED_MIN_CONDITION_PROPERTIES:-2}"
MAX_CONDITION_PROPERTIES="${SMMED_MAX_CONDITION_PROPERTIES:-7}"
IMAGE_SIZE="${SMMED_IMAGE_SIZE:-256}"
SEED="${SMMED_SEED:-7}"
RENDER_IMAGES="${SMMED_RENDER_IMAGES:-1}"

MOLECULE_DB="$OUTPUT_DIR/molecule_database.csv"
PAIR_DB="$OUTPUT_DIR/edit_pairs.csv"
CONDITION_ROWS="$OUTPUT_DIR/condition_rows.csv"
BASELINE_VARIANTS="$OUTPUT_DIR/baseline_variants.csv"
IMAGE_DIR="$OUTPUT_DIR/images"

echo "SketchMol multi-property edit dataset build"
echo "  python=$PYTHON_BIN"
echo "  input_csv=$INPUT_CSV"
echo "  output_dir=$OUTPUT_DIR"
echo "  limit=$LIMIT"
echo "  max_pairs=$MAX_PAIRS"
echo "  conditions_per_pair=$CONDITIONS_PER_PAIR"
echo "  render_images=$RENDER_IMAGES"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "ERROR: input CSV not found: $INPUT_CSV" >&2
  exit 2
fi

MOLECULE_IMAGE_ARGS=()
PAIR_IMAGE_ARGS=()
if [[ "$RENDER_IMAGES" == "1" ]]; then
  PAIR_IMAGE_ARGS=(--render-images --image-dir "$IMAGE_DIR" --image-size "$IMAGE_SIZE")
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/build_molecule_database.py" \
  --input-csv "$INPUT_CSV" \
  --output-csv "$MOLECULE_DB" \
  --limit "$LIMIT"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/build_edit_pair_database.py" \
  --molecule-db-csv "$MOLECULE_DB" \
  --output-csv "$PAIR_DB" \
  --max-pairs "$MAX_PAIRS" \
  --max-pairs-per-scaffold "$MAX_PAIRS_PER_SCAFFOLD" \
  --max-molecules-per-scaffold "$MAX_MOLS_PER_SCAFFOLD" \
  --min-active-properties "$MIN_ACTIVE_PROPERTIES" \
  --threshold-scale "$THRESHOLD_SCALE" \
  --min-similarity "$MIN_SIMILARITY" \
  --max-similarity "$MAX_SIMILARITY" \
  --eval-fraction "$EVAL_FRACTION" \
  --seed "$SEED" \
  "${PAIR_IMAGE_ARGS[@]}"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/build_condition_rows.py" \
  --edit-pairs-csv "$PAIR_DB" \
  --output-csv "$CONDITION_ROWS" \
  --baseline-variants-csv "$BASELINE_VARIANTS" \
  --conditions-per-pair "$CONDITIONS_PER_PAIR" \
  --min-properties "$MIN_CONDITION_PROPERTIES" \
  --max-properties "$MAX_CONDITION_PROPERTIES" \
  --seed "$SEED"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

out = Path("$OUTPUT_DIR")
summary = {
    "output_dir": str(out),
    "molecule_database_csv": str(Path("$MOLECULE_DB")),
    "edit_pairs_csv": str(Path("$PAIR_DB")),
    "condition_rows_csv": str(Path("$CONDITION_ROWS")),
    "baseline_variants_csv": str(Path("$BASELINE_VARIANTS")),
}
for name, path in [
    ("molecule_summary", Path("$MOLECULE_DB").with_suffix(".summary.json")),
    ("pair_summary", Path("$PAIR_DB").with_suffix(".summary.json")),
    ("condition_summary", Path("$CONDITION_ROWS").with_suffix(".summary.json")),
]:
    if path.exists():
        summary[name] = json.loads(path.read_text())
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo
echo "Multi-property edit dataset ready:"
echo "  molecule_db=$MOLECULE_DB"
echo "  edit_pairs=$PAIR_DB"
echo "  condition_rows=$CONDITION_ROWS"
echo "  baseline_variants=$BASELINE_VARIANTS"
echo "  summary=$OUTPUT_DIR/summary.json"
