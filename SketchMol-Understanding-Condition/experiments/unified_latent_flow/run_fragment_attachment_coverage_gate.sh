#!/usr/bin/env bash
# B24 evidence: train-only MMPA fragment attachment coverage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_FRAGMENT_COVERAGE_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_DIR="${SUCC_FRAGMENT_COVERAGE_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/fragment_attachment_coverage_v24/seed_1741}"
DATASET_DIR="${SUCC_FRAGMENT_COVERAGE_DATASET_DIR:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset}"

for path in \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B24 input: $path" >&2; exit 2; }
done

if [[ -f "$OUTPUT_DIR/summary.json" && "${SUCC_FRAGMENT_COVERAGE_FORCE:-0}" != "1" ]]; then
  echo "ERROR: completed B24 coverage result exists: $OUTPUT_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/audit_fragment_attachment_trajectory_coverage.py" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --output-dir "$OUTPUT_DIR" \
  --train-limit "${SUCC_FRAGMENT_COVERAGE_TRAIN_LIMIT:-1500}" \
  --validation-limit "${SUCC_FRAGMENT_COVERAGE_VALIDATION_LIMIT:-20}" \
  --property-counts "${SUCC_FRAGMENT_COVERAGE_PROPERTY_COUNTS:-2,3}" \
  --validation-exclusion-seed "${SUCC_FRAGMENT_COVERAGE_EXCLUSION_SEED:-1742}" \
  --validation-selection-seed "${SUCC_FRAGMENT_COVERAGE_VALIDATION_SEED:-2719}" \
  --train-selection-seed "${SUCC_FRAGMENT_COVERAGE_TRAIN_SEED:-1741}" \
  --min-core-heavy-atoms 5 \
  --max-variable-heavy-atoms 30 \
  --gate-overall-coverage 0.30 \
  --gate-three-property-coverage 0.30 \
  --gate-growth-task-coverage 0.30 \
  --gate-exact-reconstruction 0.95 \
  --gate-unique-target-fragments 100
