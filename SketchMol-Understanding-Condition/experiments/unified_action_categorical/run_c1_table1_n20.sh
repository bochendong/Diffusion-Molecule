#!/usr/bin/env bash
# C1: frozen graph+fragment mixture categorical, honest Table1 n=20, no ranking.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
WORKTREE_LATENT="${SUCC_LATENT_DIR:-$PROJECT_DIR/experiments/unified_latent_flow}"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-$REPO_DIR}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
OUTPUT_DIR="${SUCC_C1_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/joint_graph_fragment_categorical_c1}"
EVAL_CSV="${SUCC_C1_EVAL_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_test_rows.csv}"
GRAPH_CANDIDATES="${SUCC_C1_GRAPH_CANDIDATES:-$SHARED_PROJECT_DIR/outputs/umtp_graph_action_full_eval_v1/seed_7/eval/action/table1/candidate_pool/graph_action_candidates.csv}"
B31_DIR="${SUCC_B31_DIR:-$SHARED_PROJECT_DIR/outputs/assay_joint_site_token_latent_v31/seed_1931}"
REPRESENTATION_DIR="${SUCC_ASSAY_JOINT_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"
FRAGMENT_DIR="${SUCC_ASSAY_JOINT_FRAGMENT_DIR:-$SHARED_PROJECT_DIR/outputs/latent_fragment_attachment_kernel_v24/cpu_seed_1761}"
DEVICE="${SUCC_DEVICE:-cpu}"
EVAL_LIMIT="${SUCC_C1_EVAL_LIMIT:-0}"

for path in \
  "$EVAL_CSV" \
  "$GRAPH_CANDIDATES" \
  "$B31_DIR/assay_joint_site_token_energy.pt" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  "$WORKTREE_LATENT/assay_joint_site_token_latent_v31_preregistration.json" \
  "$SCRIPT_DIR/joint_graph_fragment_categorical_c1_preregistration.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$WORKTREE_LATENT:$WORKTREE_LATENT/../..:$WORKTREE_LATENT/../../experiments/unified_constraint_agent:$PROJECT_DIR:$REPO_DIR/SketchMol-Unified-3MDiffusion:$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$SHARED_PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"

echo "C1 joint graph+fragment categorical n=20"
echo "  python=$PYTHON_BIN"
echo "  device=$DEVICE"
echo "  eval_csv=$EVAL_CSV"
echo "  output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/eval_joint_graph_fragment_categorical_c1.py" \
  --eval-csv "$EVAL_CSV" \
  --graph-candidate-csv "$GRAPH_CANDIDATES" \
  --b31-checkpoint "$B31_DIR/assay_joint_site_token_energy.pt" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --fragment-checkpoint "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  --b31-protocol-manifest "$WORKTREE_LATENT/assay_joint_site_token_latent_v31_preregistration.json" \
  --c1-protocol-manifest "$SCRIPT_DIR/joint_graph_fragment_categorical_c1_preregistration.json" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --eval-limit "$EVAL_LIMIT"

"$PYTHON_BIN" "$SHARED_PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$EVAL_CSV" \
  --candidates "$OUTPUT_DIR/c1_table1_n20_candidates.csv" \
  --output-dir "$OUTPUT_DIR/moledit_table_metrics_any20" \
  --candidate-limit 20 \
  --model-name joint_graph_fragment_categorical_c1 \
  --task-filter table1 \
  --missing-oracle-policy fail

"$PYTHON_BIN" "$SCRIPT_DIR/collect_c1_table1_n20.py" \
  --sampling-summary "$OUTPUT_DIR/sampling_summary.json" \
  --metrics-json "$OUTPUT_DIR/moledit_table_metrics_any20/moledit_table_summary.json" \
  --preregistration "$SCRIPT_DIR/joint_graph_fragment_categorical_c1_preregistration.json" \
  --output-json "$OUTPUT_DIR/summary.json"

echo "summary=$OUTPUT_DIR/summary.json"
