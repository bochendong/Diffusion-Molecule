#!/usr/bin/env bash
# Score frozen B41/canonical/D3 Table1 candidates on one common any@k grid.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_PAPER_AUDIT_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_ROOT="${SUCC_ANYK_ROBUSTNESS_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/table1_anyk_robustness_v1}"
REFERENCE="$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_test_rows.csv"
B41="$SHARED_PROJECT_DIR/outputs/d0_b41_table1_n20/d0_b41_table1_n20_candidates.csv"
CANONICAL="$SHARED_PROJECT_DIR/outputs/b41_frontier_objective_table1_n20/canonical/b41_canonical_table1_n20_candidates.csv"
D3="$SHARED_PROJECT_DIR/outputs/d3_event_kernel_energy_grpo_table1_n20/d3_event_kernel_energy_table1_n20_candidates.csv"

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

for path in "$REFERENCE" "$B41" "$CANONICAL" "$D3" "$SUCC_GSK3B_ORACLE_PATH"; do
  [[ -f "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 2; }
done

score() {
  local name="$1"
  local candidates="$2"
  if [[ -s "$OUTPUT_ROOT/$name/anyk_curve.json" ]]; then
    echo "reuse_existing_curve=$OUTPUT_ROOT/$name/anyk_curve.json"
    return 0
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/collect_anyk_budget.py" \
    --reference "$REFERENCE" \
    --candidates "$candidates" \
    --output-dir "$OUTPUT_ROOT/$name" \
    --model-name "$name" \
    --task-filter table1 \
    --missing-oracle-policy fail
}

score b41 "$B41"
score canonical "$CANONICAL"
score d3_grpo "$D3"

"$PYTHON_BIN" "$SCRIPT_DIR/collect_table1_anyk_robustness.py" \
  --b41-curve "$OUTPUT_ROOT/b41/anyk_curve.json" \
  --canonical-curve "$OUTPUT_ROOT/canonical/anyk_curve.json" \
  --d3-curve "$OUTPUT_ROOT/d3_grpo/anyk_curve.json" \
  --b41-candidates "$B41" \
  --canonical-candidates "$CANONICAL" \
  --d3-candidates "$D3" \
  --output-json "$OUTPUT_ROOT/summary.json"

echo "summary=$OUTPUT_ROOT/summary.json"
