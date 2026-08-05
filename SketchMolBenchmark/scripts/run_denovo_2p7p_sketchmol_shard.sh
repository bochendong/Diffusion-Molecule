#!/usr/bin/env bash
# Run SketchMol + MolScribe OCR for one de novo 2p-7p shard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../../SketchMol-Understanding-Condition/scripts/molscribe_env.sh
source "$REPO_ROOT/SketchMol-Understanding-Condition/scripts/molscribe_env.sh"

# Compute nodes often lack `module` until the CC profile is sourced.
# Match resume_real_sketchmol_ocr.sh / README: OpenCV comes from the cluster module.
SKETCHMOL_MODULES="${SKETCHMOL_MODULES:-gcc opencv/4.13.0 rdkit/2024.09.6}"
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    # shellcheck source=/dev/null
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  fi
fi
if command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHMOL_MODULES
fi

SKETCHMOL_DENOVO_EVAL_CSV="${SKETCHMOL_DENOVO_EVAL_CSV:-SketchMol-Understanding-Condition/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_eval_rows.csv}"
SKETCHMOL_DENOVO_OUTPUT_DIR="${SKETCHMOL_DENOVO_OUTPUT_DIR:-SketchMolBenchmark/outputs/sketchmol_denovo_2p7p_at40_v1}"
SKETCHMOL_DENOVO_SHARD_COUNT="${SKETCHMOL_DENOVO_SHARD_COUNT:-600}"
SKETCHMOL_DENOVO_SHARD_INDEX="${SKETCHMOL_DENOVO_SHARD_INDEX:?SKETCHMOL_DENOVO_SHARD_INDEX is required}"
SKETCHMOL_DENOVO_RESUME="${SKETCHMOL_DENOVO_RESUME:-1}"
# Prefer the denovo-specific var; allow SKETCHMOL_CONDITIONAL_COUNT as a fallback
# so callers that only set the generic name still get the intended budget.
SKETCHMOL_DENOVO_CONDITIONAL_COUNT="${SKETCHMOL_DENOVO_CONDITIONAL_COUNT:-${SKETCHMOL_CONDITIONAL_COUNT:-40}}"
SKETCHMOL_DENOVO_SAMPLE_BATCH_SIZE="${SKETCHMOL_DENOVO_SAMPLE_BATCH_SIZE:-$SKETCHMOL_DENOVO_CONDITIONAL_COUNT}"
SKETCHMOL_DENOVO_CUSTOM_STEPS="${SKETCHMOL_DENOVO_CUSTOM_STEPS:-250}"
SKETCHMOL_DENOVO_SCALE="${SKETCHMOL_DENOVO_SCALE:-1.2}"
SKETCHMOL_DENOVO_SCALE_PRO="${SKETCHMOL_DENOVO_SCALE_PRO:-6.3}"

SKETCHMOL_REPO="${SKETCHMOL_REPO:-Research/Molecule Generation/SketchMol/SketchMol-v1-main}"
SKETCHMOL_PYTHON_BIN="${SKETCHMOL_PYTHON_BIN:-/scratch/bdong/venvs/phystabmol/bin/python}"
# README default for MolScribe OCR is molscribe_overlay (has OpenCV + RDKit pathing).
SKETCHMOL_MOLSCRIBE_PYTHON_BIN="${SKETCHMOL_MOLSCRIBE_PYTHON_BIN:-/home/bdong/.venvs/molscribe_torch251/bin/python}"
SKETCHMOL_MOLSCRIBE_SCRIPT="${SKETCHMOL_MOLSCRIBE_SCRIPT:-SketchMol-Understanding-Condition/scripts/run_molscribe_ocr.py}"
SKETCHMOL_CKPT="${SKETCHMOL_CKPT:-/scratch/bdong/checkpoints/sketchmol/model_weights.ckpt}"
SKETCHMOL_MOLSCRIBE_MODEL="${SKETCHMOL_MOLSCRIBE_MODEL:-/scratch/bdong/checkpoints/molscribe/swin_base_char_aux_200k.pth}"
SKETCHMOL_MOLSCRIBE_BATCH_SIZE="${SKETCHMOL_MOLSCRIBE_BATCH_SIZE:-8}"
SKETCHMOL_MOLSCRIBE_BACKEND="${SKETCHMOL_MOLSCRIBE_BACKEND:-custom}"
SKETCHMOL_MOLSCRIBE_WORKDIR="${SKETCHMOL_MOLSCRIBE_WORKDIR:-Research/Molecule Generation/SketchMol/SketchMol-v1-main/evaluate}"
SKETCHMOL_ONMT_OVERLAY="${SKETCHMOL_ONMT_OVERLAY:-/scratch/bdong/python_overlays/onmt220}"
SKETCHMOL_MOLSCRIBE_PIP_OVERLAY="${SKETCHMOL_MOLSCRIBE_PIP_OVERLAY:-/scratch/bdong/python_overlays/molscribe111}"
export SKETCHMOL_MOLSCRIBE_WORKDIR SKETCHMOL_ONMT_OVERLAY
export SUCC_MOLSCRIBE_WORKDIR="${SUCC_MOLSCRIBE_WORKDIR:-$SKETCHMOL_MOLSCRIBE_WORKDIR}"
export SUCC_ONMT_OVERLAY="${SUCC_ONMT_OVERLAY:-$SKETCHMOL_ONMT_OVERLAY}"
export SUCC_PREFER_PIP_MOLSCRIBE="${SUCC_PREFER_PIP_MOLSCRIBE:-1}"
export SUCC_MOLSCRIBE_PIP_OVERLAY="${SUCC_MOLSCRIBE_PIP_OVERLAY:-$SKETCHMOL_MOLSCRIBE_PIP_OVERLAY}"

prepend_molscribe_pythonpath

echo "SketchMol de novo 2p-7p shard"
echo "  eval_csv=$SKETCHMOL_DENOVO_EVAL_CSV"
echo "  output_dir=$SKETCHMOL_DENOVO_OUTPUT_DIR"
echo "  shard_index=$SKETCHMOL_DENOVO_SHARD_INDEX"
echo "  shard_count=$SKETCHMOL_DENOVO_SHARD_COUNT"
echo "  conditional_count=$SKETCHMOL_DENOVO_CONDITIONAL_COUNT"
echo "  sample_batch_size=$SKETCHMOL_DENOVO_SAMPLE_BATCH_SIZE"
echo "  sketchmol_python=$SKETCHMOL_PYTHON_BIN"
echo "  molscribe_python=$SKETCHMOL_MOLSCRIBE_PYTHON_BIN"
echo "  modules=$SKETCHMOL_MODULES"
echo "  onmt_overlay=$SKETCHMOL_ONMT_OVERLAY"
echo "  molscribe_pip_overlay=$SKETCHMOL_MOLSCRIBE_PIP_OVERLAY"
echo "  molscribe_workdir=$SKETCHMOL_MOLSCRIBE_WORKDIR"

RESUME_FLAG=()
if [[ "$SKETCHMOL_DENOVO_RESUME" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

"$SKETCHMOL_PYTHON_BIN" "$SCRIPT_DIR/denovo_2p7p_sketchmol_shard.py" \
  --eval-csv "$SKETCHMOL_DENOVO_EVAL_CSV" \
  --output-dir "$SKETCHMOL_DENOVO_OUTPUT_DIR" \
  --shard-index "$SKETCHMOL_DENOVO_SHARD_INDEX" \
  --shard-count "$SKETCHMOL_DENOVO_SHARD_COUNT" \
  "${RESUME_FLAG[@]}" \
  --sketchmol-repo "$SKETCHMOL_REPO" \
  --sketchmol-python "$SKETCHMOL_PYTHON_BIN" \
  --molscribe-python "$SKETCHMOL_MOLSCRIBE_PYTHON_BIN" \
  --molscribe-script "$SKETCHMOL_MOLSCRIBE_SCRIPT" \
  --ckpt "$SKETCHMOL_CKPT" \
  --molscribe-model "$SKETCHMOL_MOLSCRIBE_MODEL" \
  --conditional-count "$SKETCHMOL_DENOVO_CONDITIONAL_COUNT" \
  --sample-batch-size "$SKETCHMOL_DENOVO_SAMPLE_BATCH_SIZE" \
  --custom-steps "$SKETCHMOL_DENOVO_CUSTOM_STEPS" \
  --scale "$SKETCHMOL_DENOVO_SCALE" \
  --scale-pro "$SKETCHMOL_DENOVO_SCALE_PRO" \
  --molscribe-batch-size "$SKETCHMOL_MOLSCRIBE_BATCH_SIZE" \
  --molscribe-backend "$SKETCHMOL_MOLSCRIBE_BACKEND"
