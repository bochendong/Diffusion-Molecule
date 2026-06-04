#!/usr/bin/env bash
# Build a development trajectory JSONL from SketchMol opt_examples.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${LETA_PYTHON_BIN:-${PYTHON_BIN:-/home/bdong/scratch/venvs/phystabmol/bin/python}}"
MODULES="${LETA_MODULES:-gcc rdkit/2025.09.4}"
OPT_EXAMPLES_DIR="${LETA_OPT_EXAMPLES_DIR:-/home/bdong/scratch/projects/Diffusion-Molecule/Research/Molecule Generation/SketchMol/SketchMol-v1-main/opt_examples}"
OUTPUT_JSONL="${LETA_TRAJECTORY_JSONL:-outputs/trajectories/sketchmol_opt_bootstrap.jsonl}"
OUTPUT_CSV="${LETA_TRAJECTORY_CSV:-outputs/trajectories/sketchmol_opt_bootstrap.csv}"
STEPS_PER_TRAJECTORY="${LETA_STEPS_PER_TRAJECTORY:-5}"
MAX_PAIRS_PER_TASK="${LETA_MAX_PAIRS_PER_TASK:-}"

if [[ -n "$MODULES" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  -m latent_edit_trajectory_attention.trajectory_generator
  bootstrap-opt
  --opt-examples-dir "$OPT_EXAMPLES_DIR"
  --output-jsonl "$OUTPUT_JSONL"
  --output-csv "$OUTPUT_CSV"
  --steps-per-trajectory "$STEPS_PER_TRAJECTORY"
)
if [[ -n "$MAX_PAIRS_PER_TASK" ]]; then
  ARGS+=(--max-pairs-per-task "$MAX_PAIRS_PER_TASK")
fi

echo "Bootstrapping SketchMol trajectories"
echo "  python=$PYTHON_BIN"
echo "  opt_examples_dir=$OPT_EXAMPLES_DIR"
echo "  output_jsonl=$OUTPUT_JSONL"
echo "  steps_per_trajectory=$STEPS_PER_TRAJECTORY"
"$PYTHON_BIN" "${ARGS[@]}"

