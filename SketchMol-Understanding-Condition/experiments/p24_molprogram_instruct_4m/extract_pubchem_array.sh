#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P24_SCRIPT_DIR:?P24_SCRIPT_DIR must be exported}"
RAW_ROOT="${P24_PUBCHEM_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule/raw/pubchem/current_sdf}"
WORK_ROOT="${P24_WORK_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule/processed/molprogram-instruct-4m-v1}"
PY="${P24_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi
mapfile -t inputs < <(find "$RAW_ROOT" -maxdepth 1 -name 'Compound_*.sdf.gz' -type f | sort)
task="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
[[ ${#inputs[@]} -eq 8 ]] || { echo "ERROR: expected 8 SDF chunks, found ${#inputs[@]}" >&2; exit 2; }
input="${inputs[$task]}"
mkdir -p "$WORK_ROOT/candidates" "$WORK_ROOT/extract_summaries"
"$PY" "$SCRIPT_DIR/extract_pubchem.py" \
  --input-sdf-gz "$input" \
  --output-jsonl "$WORK_ROOT/candidates/chunk-$task.jsonl" \
  --summary-json "$WORK_ROOT/extract_summaries/chunk-$task.json"

