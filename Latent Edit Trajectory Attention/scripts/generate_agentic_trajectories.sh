#!/usr/bin/env bash
# Generate path-dependent RDKit-agent trajectories from SketchMol opt_examples.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${LETA_PYTHON_BIN:-${PYTHON_BIN:-/home/bdong/scratch/venvs/phystabmol/bin/python}}"
MODULES="${LETA_MODULES:-gcc rdkit/2025.09.4}"
OPT_EXAMPLES_DIR="${LETA_OPT_EXAMPLES_DIR:-/home/bdong/scratch/projects/Diffusion-Molecule/Research/Molecule Generation/SketchMol/SketchMol-v1-main/opt_examples}"
OUTPUT_JSONL="${LETA_TRAJECTORY_JSONL:-outputs/trajectories/sketchmol_agentic_opt.jsonl}"
OUTPUT_CSV="${LETA_TRAJECTORY_CSV:-outputs/trajectories/sketchmol_agentic_opt.csv}"
TRAJECTORIES_PER_TASK="${LETA_TRAJECTORIES_PER_TASK:-24}"
STEPS_PER_TRAJECTORY="${LETA_STEPS_PER_TRAJECTORY:-6}"
TOP_K="${LETA_TOP_K:-8}"
SIMILARITY_WEIGHT="${LETA_SIMILARITY_WEIGHT:-0.15}"
NOVELTY_WEIGHT="${LETA_NOVELTY_WEIGHT:-0.05}"
SEED="${LETA_SEED:-7}"

if [[ -n "$MODULES" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $MODULES
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Generating agentic SketchMol trajectories"
echo "  python=$PYTHON_BIN"
echo "  opt_examples_dir=$OPT_EXAMPLES_DIR"
echo "  output_jsonl=$OUTPUT_JSONL"
echo "  trajectories_per_task=$TRAJECTORIES_PER_TASK"
echo "  steps_per_trajectory=$STEPS_PER_TRAJECTORY"

"$PYTHON_BIN" -m latent_edit_trajectory_attention.trajectory_generator agentic-opt \
  --opt-examples-dir "$OPT_EXAMPLES_DIR" \
  --output-jsonl "$OUTPUT_JSONL" \
  --output-csv "$OUTPUT_CSV" \
  --trajectories-per-task "$TRAJECTORIES_PER_TASK" \
  --steps-per-trajectory "$STEPS_PER_TRAJECTORY" \
  --top-k "$TOP_K" \
  --similarity-weight "$SIMILARITY_WEIGHT" \
  --novelty-weight "$NOVELTY_WEIGHT" \
  --seed "$SEED"

