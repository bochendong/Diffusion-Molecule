#!/usr/bin/env bash
# Frontier vs canonical vs random-singleton event-kernel labels.
# Two 2-epoch fine-tunes from B39, then Table1 n=20. Frontier is frozen D0 B41.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C_DIR="$(cd "$SCRIPT_DIR/../unified_action_categorical" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
WORKTREE_LATENT="${SUCC_LATENT_DIR:-$PROJECT_DIR/experiments/unified_latent_flow}"
cd "$REPO_DIR"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
  module load cuda/12.6 2>/dev/null || true
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-$REPO_DIR}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
OUTPUT_DIR="${SUCC_FRONTIER_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/b41_frontier_objective_table1_n20}"
EVAL_CSV="${SUCC_D0_EVAL_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_test_rows.csv}"
DATASET_DIR="$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
B22_DIR="$SHARED_PROJECT_DIR/outputs/valid_early_stop_delta_diffusion_v22/seed_1757"
B36_DIR="$SHARED_PROJECT_DIR/outputs/source_anchored_graph_patch_evidence_v36/seed_1981"
B37_DIR="$SHARED_PROJECT_DIR/outputs/source_clamped_region_graph_diffusion_v37/seed_1983"
B38_DIR="$SHARED_PROJECT_DIR/outputs/source_clamped_latent_graph_jump_process_v38/seed_1985"
B39_DIR="$SHARED_PROJECT_DIR/outputs/latent_cardinality_graph_jump_bridge_v39/seed_1987"
B40_DIR="$SHARED_PROJECT_DIR/outputs/valence_constrained_latent_particle_bridge_v40/seed_1989"
B41_DIR="$SHARED_PROJECT_DIR/outputs/viability_preserving_interacting_particle_transport_v41/seed_1991"
B41_TABLE1="$SHARED_PROJECT_DIR/outputs/d0_b41_table1_n20"
DEVICE="${SUCC_FRONTIER_DEVICE:-auto}"
EVAL_LIMIT="${SUCC_D0_EVAL_LIMIT:-0}"
PROTOCOL="$SCRIPT_DIR/b41_frontier_objective_preregistration.json"

for path in \
  "$EVAL_CSV" \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  "$B22_DIR/summary.json" \
  "$B36_DIR/summary.json" \
  "$B37_DIR/summary.json" \
  "$B38_DIR/source_clamped_latent_graph_jump_process.pt" \
  "$B38_DIR/summary.json" \
  "$B39_DIR/latent_cardinality_graph_jump_bridge.pt" \
  "$B39_DIR/summary.json" \
  "$B39_DIR/evaluated_train_only_dev_candidates.csv" \
  "$B40_DIR/summary.json" \
  "$B40_DIR/evaluated_train_only_dev_candidates.csv" \
  "$B41_DIR/viability_interacting_particle_transport.pt" \
  "$B41_DIR/summary.json" \
  "$B41_DIR/evaluated_train_only_dev_candidates.csv" \
  "$B41_TABLE1/summary.json" \
  "$WORKTREE_LATENT/viability_preserving_interacting_particle_transport_v41_preregistration.json" \
  "$PROTOCOL"; do
  [[ -f "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$WORKTREE_LATENT:$WORKTREE_LATENT/../..:$WORKTREE_LATENT/../../experiments/unified_constraint_agent:$PROJECT_DIR:$C_DIR:$SCRIPT_DIR:$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$SHARED_PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"

echo "B41 frontier objective Table1 n=20"
echo "  python=$PYTHON_BIN"
echo "  device=$DEVICE"
echo "  output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/eval_b41_frontier_objective.py" \
  --eval-csv "$EVAL_CSV" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --b22-checkpoint "$B22_DIR/valid_early_stop_delta_diffusion.pt" \
  --b22-summary "$B22_DIR/summary.json" \
  --b36-summary "$B36_DIR/summary.json" \
  --b37-summary "$B37_DIR/summary.json" \
  --b38-checkpoint "$B38_DIR/source_clamped_latent_graph_jump_process.pt" \
  --b38-summary "$B38_DIR/summary.json" \
  --b39-checkpoint "$B39_DIR/latent_cardinality_graph_jump_bridge.pt" \
  --b39-summary "$B39_DIR/summary.json" \
  --b39-evaluated-candidates "$B39_DIR/evaluated_train_only_dev_candidates.csv" \
  --b40-summary "$B40_DIR/summary.json" \
  --b40-evaluated-candidates "$B40_DIR/evaluated_train_only_dev_candidates.csv" \
  --b41-checkpoint "$B41_DIR/viability_interacting_particle_transport.pt" \
  --b41-summary "$B41_DIR/summary.json" \
  --b41-evaluated-candidates "$B41_DIR/evaluated_train_only_dev_candidates.csv" \
  --b41-protocol-manifest "$WORKTREE_LATENT/viability_preserving_interacting_particle_transport_v41_preregistration.json" \
  --audit-protocol-manifest "$PROTOCOL" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --eval-limit "$EVAL_LIMIT"

for VARIANT in canonical random_singleton; do
  "$PYTHON_BIN" "$SHARED_PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
    --reference "$EVAL_CSV" \
    --candidates "$OUTPUT_DIR/$VARIANT/b41_${VARIANT}_table1_n20_candidates.csv" \
    --output-dir "$OUTPUT_DIR/$VARIANT/moledit_table_metrics_any20" \
    --candidate-limit 20 \
    --model-name "b41_${VARIANT}" \
    --task-filter table1 \
    --missing-oracle-policy fail

  "$PYTHON_BIN" "$C_DIR/collect_c1_table1_n20.py" \
    --sampling-summary "$OUTPUT_DIR/$VARIANT/sampling_summary.json" \
    --metrics-json "$OUTPUT_DIR/$VARIANT/moledit_table_metrics_any20/moledit_table_summary.json" \
    --preregistration "$PROTOCOL" \
    --output-json "$OUTPUT_DIR/$VARIANT/summary.json"
done

"$PYTHON_BIN" "$SCRIPT_DIR/collect_b41_frontier_objective.py" \
  --preregistration "$PROTOCOL" \
  --frontier-summary "$B41_TABLE1/summary.json" \
  --canonical-summary "$OUTPUT_DIR/canonical/summary.json" \
  --random-summary "$OUTPUT_DIR/random_singleton/summary.json" \
  --output-json "$OUTPUT_DIR/summary.json"

echo "summary=$OUTPUT_DIR/summary.json"
