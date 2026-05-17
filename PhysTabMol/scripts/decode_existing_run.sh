#!/bin/bash
#SBATCH --job-name=phystabmol-decode
#SBATCH --account=def-hup-ab
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=./decode-%j.log
#SBATCH --error=./decode-%j.log

set -euo pipefail

# Use this script after a training job already wrote:
#   runs/<run>/tables/generated_table_rows.csv
#
# Direct run inside an allocation:
#   bash scripts/decode_existing_run.sh
#   bash scripts/decode_existing_run.sh runs/20260501_145202_slurm_13110897
#
# Or submit as a CPU Slurm job:
#   sbatch scripts/decode_existing_run.sh
#   PHYSTABMOL_RUN_DIR=runs/20260501_145202_slurm_13110897 sbatch scripts/decode_existing_run.sh

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$SLURM_SUBMIT_DIR}"
else
  PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
SCRIPT_DIR="$PHYSTABMOL_ROOT/scripts"
cd "$PHYSTABMOL_ROOT"
export PHYSTABMOL_SUPPRESS_RDKIT_LOGS="${PHYSTABMOL_SUPPRESS_RDKIT_LOGS:-1}"
export PHYSTABMOL_PROGRESS="${PHYSTABMOL_PROGRESS:-1}"
export PHYSTABMOL_PROGRESS_STEP="${PHYSTABMOL_PROGRESS_STEP:-5}"

if [[ "${PHYSTABMOL_SKIP_ENV:-0}" != "1" ]]; then
  if command -v module >/dev/null 2>&1 && [[ -f "$SCRIPT_DIR/env_module_venv.sh" ]]; then
    export PHYSTABMOL_ROOT
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/env_module_venv.sh"
  elif [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
  else
    echo "No module command or .venv found; using current Python environment."
  fi
fi

DEFAULT_RUN_DIR="runs/20260505_002855_sketchmol_comparable_structure_v1"
RUN_DIR="${PHYSTABMOL_RUN_DIR:-${1:-$DEFAULT_RUN_DIR}}"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(
    find runs -path 'runs/*/tables/generated_table_rows.csv' -print 2>/dev/null \
      | sed 's#/tables/generated_table_rows.csv##' \
      | while read -r candidate; do
          if [[ ! -s "$candidate/tables/decoded_candidates.csv" ]]; then
            echo "$candidate"
          fi
        done \
      | sort \
      | tail -1
  )"
fi
if [[ -z "$RUN_DIR" ]]; then
  echo "Could not infer a run directory. Pass one explicitly:" >&2
  echo "  bash scripts/decode_existing_run.sh runs/<run_name>" >&2
  exit 2
fi

MAX_CONDITIONS="${PHYSTABMOL_MAX_CONDITIONS:-5000}"
SAMPLES_PER_CONDITION="${PHYSTABMOL_DECODE_SAMPLES:-8}"
DECODE_TOP_K="${PHYSTABMOL_DECODE_TOP_K:-2}"
MAX_ROWS="${PHYSTABMOL_MAX_ROWS:-0}"
OUTPUT_NAME="${PHYSTABMOL_OUTPUT_NAME:-decoded_candidates.csv}"
ENABLE_3D="${PHYSTABMOL_ENABLE_3D:-0}"
SAVE_3D_SDF="${PHYSTABMOL_SAVE_3D_SDF:-0}"
DYNAMIC_DECODER="${PHYSTABMOL_DYNAMIC_DECODER:-0}"

EXTRA_ARGS=()
if [[ "$DYNAMIC_DECODER" == "1" ]]; then
  EXTRA_ARGS+=(--dynamic-decoder)
fi
if [[ "$ENABLE_3D" == "1" ]]; then
  EXTRA_ARGS+=(--enable-3d)
fi
if [[ "$SAVE_3D_SDF" == "1" ]]; then
  EXTRA_ARGS+=(--save-3d-sdf)
fi

echo "Decoding run: $RUN_DIR"
echo "max_conditions=$MAX_CONDITIONS samples_per_condition=$SAMPLES_PER_CONDITION decode_top_k=$DECODE_TOP_K"
echo "output_name=$OUTPUT_NAME dynamic_decoder=$DYNAMIC_DECODER enable_3d=$ENABLE_3D"

python3 -m phystabmol.decode_run \
  --run-dir "$RUN_DIR" \
  --max-conditions "$MAX_CONDITIONS" \
  --samples-per-condition "$SAMPLES_PER_CONDITION" \
  --max-rows "$MAX_ROWS" \
  --decode-top-k "$DECODE_TOP_K" \
  --output-name "$OUTPUT_NAME" \
  "${EXTRA_ARGS[@]}"

echo "Decode finished at $(date -Is)"
echo "Summary:"
cat "$RUN_DIR/summary.txt"
