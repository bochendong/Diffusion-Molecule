#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${P24_SCRIPT_DIR:?P24_SCRIPT_DIR must be exported}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_ROOT="${P24_WORK_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule/processed/molprogram-instruct-4m-v1}"
EDIT_ROOT="${P24_EDIT_ROOT:-/scratch/bdong/datasets/Diffusion-Molecule/processed/moledit-instruct/enhanced_v1/splits}"
PY="${P24_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023 python/3.11 rdkit/2025.09.4
fi
export PYTHONPATH="$SCRIPT_DIR:$PROJECT/experiments/p23_explicit_task_stage1_v2${PYTHONPATH:+:$PYTHONPATH}"
release="$WORK_ROOT/release"
mkdir -p "$release/manifests"
inputs=()
for path in "$WORK_ROOT"/candidates/chunk-*.jsonl; do
  inputs+=(--input "$path")
done
[[ ${#inputs[@]} -eq 16 ]] || { echo "ERROR: expected 8 candidate JSONL files" >&2; exit 2; }

"$PY" "$SCRIPT_DIR/build_release.py" de_novo \
  "${inputs[@]}" --output-dir "$release" --target-rows 2000000 --shards 128 \
  --heldout "$PROJECT/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition/denovo_2p7p_eval_rows.csv" \
  --manifest "$release/manifests/de_novo.json"
"$PY" "$SCRIPT_DIR/build_release.py" edit \
  --input-csv "$EDIT_ROOT/train.csv" --output-dir "$release" --target-rows 569919 --shards 128 \
  --heldout "$EDIT_ROOT/eval_balanced.csv" --manifest "$release/manifests/edit.json"
"$PY" "$SCRIPT_DIR/finalize_release.py" \
  --release-root "$release" --de-novo-rows 2000000 --edit-rows 569919 \
  --output-manifest "$release/MolProgramInstruct-Balanced-v1.manifest.json"
echo "P24 release complete: $release"
