#!/usr/bin/env bash
# D2: frozen graph AE + source-ball short rectified flow, Table1 n=20.

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
OUTPUT_DIR="${SUCC_D2_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/d2_source_ball_flow_table1_n20}"
EVAL_CSV="${SUCC_D0_EVAL_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_test_rows.csv}"
DATASET_DIR="$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset"
OFFICIAL_GSK3B="${SUCC_D0_OFFICIAL_GSK3B_CSV:-$SHARED_PROJECT_DIR/outputs/direct_smiles_moledit_table1_gsk3b_n20_pilot_v1/gsk3b_pack/table1_eval_gsk3b_moledit_rows.csv}"
REPRESENTATION_DIR="$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725"
DEVICE="${SUCC_D2_DEVICE:-auto}"
EVAL_LIMIT="${SUCC_D0_EVAL_LIMIT:-0}"

for path in \
  "$EVAL_CSV" \
  "$DATASET_DIR/unified_joint_train_rows.csv" \
  "$DATASET_DIR/unified_joint_validation_rows.csv" \
  "$OFFICIAL_GSK3B" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$SCRIPT_DIR/d2_source_ball_flow_preregistration.json" \
  "$SCRIPT_DIR/eval_d2_source_ball_flow.py" \
  "$WORKTREE_LATENT/categorical_graph_latent_flow.py"; do
  [[ -f "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$WORKTREE_LATENT:$WORKTREE_LATENT/../..:$PROJECT_DIR:$C_DIR:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$SHARED_PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"

echo "D2 source-ball latent flow Table1 n=20"
echo "  python=$PYTHON_BIN"
echo "  device=$DEVICE"
echo "  output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/eval_d2_source_ball_flow.py" \
  --eval-csv "$EVAL_CSV" \
  --train-csv "$DATASET_DIR/unified_joint_train_rows.csv" \
  --validation-csv "$DATASET_DIR/unified_joint_validation_rows.csv" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --d2-protocol-manifest "$SCRIPT_DIR/d2_source_ball_flow_preregistration.json" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --eval-limit "$EVAL_LIMIT"

"$PYTHON_BIN" "$SHARED_PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$EVAL_CSV" \
  --candidates "$OUTPUT_DIR/d2_source_ball_table1_n20_candidates.csv" \
  --output-dir "$OUTPUT_DIR/moledit_table_metrics_any20" \
  --candidate-limit 20 \
  --model-name d2_source_ball_latent_flow \
  --task-filter table1 \
  --missing-oracle-policy fail

"$PYTHON_BIN" "$C_DIR/align_official_gsk3b_candidates.py" \
  --official-reference "$OFFICIAL_GSK3B" \
  --candidates "$OUTPUT_DIR/d2_source_ball_table1_n20_candidates.csv" \
  --output-csv "$OUTPUT_DIR/d2_official_gsk3b_n20_candidates.csv"

"$PYTHON_BIN" "$SHARED_PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$OFFICIAL_GSK3B" \
  --candidates "$OUTPUT_DIR/d2_official_gsk3b_n20_candidates.csv" \
  --output-dir "$OUTPUT_DIR/official_gsk3b_metrics_any20" \
  --candidate-limit 20 \
  --model-name d2_source_ball_latent_flow_official_gsk3b \
  --task-filter table1 \
  --missing-oracle-policy fail

"$PYTHON_BIN" "$C_DIR/collect_c1_table1_n20.py" \
  --sampling-summary "$OUTPUT_DIR/sampling_summary.json" \
  --metrics-json "$OUTPUT_DIR/moledit_table_metrics_any20/moledit_table_summary.json" \
  --preregistration "$SCRIPT_DIR/d2_source_ball_flow_preregistration.json" \
  --official-gsk3b-json "$OUTPUT_DIR/official_gsk3b_metrics_any20/moledit_table_summary.json" \
  --output-json "$OUTPUT_DIR/summary.json"

echo "summary=$OUTPUT_DIR/summary.json"
