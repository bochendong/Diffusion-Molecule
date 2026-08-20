#!/usr/bin/env bash
# Finish oracle scoring for already-generated particle candidates. No generation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_PAPER_AUDIT_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_ROOT="${SUCC_PARTICLE_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/b41_particle_coverage_table1_n20}"
REFERENCE="$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_test_rows.csv"
PREREGISTRATION="$SCRIPT_DIR/b41_particle_coverage_preregistration.json"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$SHARED_REPO_DIR/SketchMol-Unified-3MDiffusion/scripts${PYTHONPATH:+:$PYTHONPATH}"
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$SHARED_PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

for path in "$REFERENCE" "$PREREGISTRATION" "$SUCC_GSK3B_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 2; }
done

for variant in iid_independent ortho_independent iid_interacting; do
  candidates="$OUTPUT_ROOT/$variant/b41_${variant}_table1_n20_candidates.csv"
  [[ -f "$candidates" ]] || { echo "ERROR: missing frozen candidates: $candidates" >&2; exit 2; }
  "$PYTHON_BIN" "$SCRIPT_DIR/collect_anyk_budget.py" \
    --reference "$REFERENCE" \
    --candidates "$candidates" \
    --output-dir "$OUTPUT_ROOT/$variant/moledit_table_metrics_anyk" \
    --model-name "b41_${variant}" \
    --task-filter table1 \
    --missing-oracle-policy fail
done

full_curve="$OUTPUT_ROOT/full_interacting/moledit_table_metrics_anyk/anyk_curve.json"
[[ -f "$full_curve" ]] || { echo "ERROR: missing existing full curve: $full_curve" >&2; exit 2; }

"$PYTHON_BIN" "$SCRIPT_DIR/collect_b41_particle_coverage.py" \
  --preregistration "$PREREGISTRATION" \
  --full-curve "$full_curve" \
  --iid-independent-curve "$OUTPUT_ROOT/iid_independent/moledit_table_metrics_anyk/anyk_curve.json" \
  --ortho-independent-curve "$OUTPUT_ROOT/ortho_independent/moledit_table_metrics_anyk/anyk_curve.json" \
  --iid-interacting-curve "$OUTPUT_ROOT/iid_interacting/moledit_table_metrics_anyk/anyk_curve.json" \
  --output-json "$OUTPUT_ROOT/summary.json"

echo "summary=$OUTPUT_ROOT/summary.json"
