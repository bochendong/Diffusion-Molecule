#!/usr/bin/env bash
# C5: GRPO on GraphEditDSL scorer, rescore C1 pool, honest Table1 n=20 mixture.

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
  module load cuda/12.6 2>/dev/null || true
fi

PYTHON_BIN="${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-$REPO_DIR}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
OUTPUT_DIR="${SUCC_C5_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/joint_graph_fragment_categorical_c5}"
EVAL_CSV="${SUCC_C5_EVAL_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_test_rows.csv}"
TRAIN_CSV="${SUCC_C5_TRAIN_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_validation_rows.csv}"
GRAPH_CANDIDATES="${SUCC_C5_GRAPH_CANDIDATES:-$SHARED_PROJECT_DIR/outputs/umtp_graph_action_full_eval_v1/seed_7/eval/action/table1/candidate_pool/graph_action_candidates.csv}"
OFFICIAL_GSK3B="${SUCC_C5_OFFICIAL_GSK3B_CSV:-$SHARED_PROJECT_DIR/outputs/direct_smiles_moledit_table1_gsk3b_n20_pilot_v1/gsk3b_pack/table1_eval_gsk3b_moledit_rows.csv}"
BASE_CHECKPOINT="${SUCC_C5_BASE_CHECKPOINT:-$SHARED_PROJECT_DIR/outputs/umtp_graph_action_protected_pilot_v1/seed_7/action_policy/umtp_graph_action_policy.pt}"
TRAIN_FEATURES="${SUCC_C5_TRAIN_FEATURES:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/feature_variants/validation_condition_features_hf_vlm}"
EVAL_FEATURES="${SUCC_C5_EVAL_FEATURES:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_suite_v1/feature_variants/eval_condition_features_hf_vlm}"
B31_DIR="${SUCC_B31_DIR:-$SHARED_PROJECT_DIR/outputs/assay_joint_site_token_latent_v31/seed_1931}"
REPRESENTATION_DIR="${SUCC_ASSAY_JOINT_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"
FRAGMENT_DIR="${SUCC_ASSAY_JOINT_FRAGMENT_DIR:-$SHARED_PROJECT_DIR/outputs/latent_fragment_attachment_kernel_v24/cpu_seed_1761}"
TRAIN_DEVICE="${SUCC_C5_TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${SUCC_DEVICE:-cpu}"
EVAL_LIMIT="${SUCC_C5_EVAL_LIMIT:-0}"
TRAIN_LIMIT="${SUCC_C5_TRAIN_LIMIT:-0}"
SCORE_BATCH="${SUCC_C5_SCORE_BATCH:-64}"

for path in \
  "$EVAL_CSV" \
  "$TRAIN_CSV" \
  "$GRAPH_CANDIDATES" \
  "$OFFICIAL_GSK3B" \
  "$BASE_CHECKPOINT" \
  "$B31_DIR/assay_joint_site_token_energy.pt" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  "$WORKTREE_LATENT/assay_joint_site_token_latent_v31_preregistration.json" \
  "$SCRIPT_DIR/joint_graph_fragment_categorical_c5_preregistration.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 2; }
done
for path in "$TRAIN_FEATURES" "$EVAL_FEATURES"; do
  [[ -d "$path" ]] || { echo "ERROR: missing feature dir: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator:$PROJECT_DIR/scripts:$WORKTREE_LATENT:$WORKTREE_LATENT/../..:$WORKTREE_LATENT/../../experiments/unified_constraint_agent:$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts:$PROJECT_DIR:$REPO_DIR/SketchMol-Unified-3MDiffusion${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$SHARED_PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"

echo "C5 graph GRPO + C1 mixture n=20"
echo "  python=$PYTHON_BIN"
echo "  train_device=$TRAIN_DEVICE"
echo "  score_batch=$SCORE_BATCH"
echo "  output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/train_graph_action_grpo_c5.py" \
  --c5-protocol-manifest "$SCRIPT_DIR/joint_graph_fragment_categorical_c5_preregistration.json" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --train-csv "$TRAIN_CSV" \
  --eval-csv "$EVAL_CSV" \
  --train-features-dir "$TRAIN_FEATURES" \
  --eval-features-dir "$EVAL_FEATURES" \
  --graph-candidate-csv "$GRAPH_CANDIDATES" \
  --output-dir "$OUTPUT_DIR" \
  --device "$TRAIN_DEVICE" \
  --train-limit "$TRAIN_LIMIT" \
  --score-batch-size "$SCORE_BATCH"

RESCORDED="$OUTPUT_DIR/graph_action_candidates_rescored.csv"
[[ -f "$RESCORDED" ]] || { echo "ERROR: missing rescored candidates: $RESCORDED" >&2; exit 2; }

"$PYTHON_BIN" "$SCRIPT_DIR/eval_joint_graph_fragment_categorical_c1.py" \
  --eval-csv "$EVAL_CSV" \
  --graph-candidate-csv "$RESCORDED" \
  --b31-checkpoint "$B31_DIR/assay_joint_site_token_energy.pt" \
  --representation-checkpoint "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  --representation-summary "$REPRESENTATION_DIR/summary.json" \
  --fragment-checkpoint "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  --b31-protocol-manifest "$WORKTREE_LATENT/assay_joint_site_token_latent_v31_preregistration.json" \
  --c1-protocol-manifest "$SCRIPT_DIR/joint_graph_fragment_categorical_c5_preregistration.json" \
  --output-dir "$OUTPUT_DIR" \
  --candidate-output "$OUTPUT_DIR/c5_table1_n20_candidates.csv" \
  --device "$EVAL_DEVICE" \
  --eval-limit "$EVAL_LIMIT"

"$PYTHON_BIN" "$SHARED_PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$EVAL_CSV" \
  --candidates "$OUTPUT_DIR/c5_table1_n20_candidates.csv" \
  --output-dir "$OUTPUT_DIR/moledit_table_metrics_any20" \
  --candidate-limit 20 \
  --model-name joint_graph_fragment_categorical_c5 \
  --task-filter table1 \
  --missing-oracle-policy fail

"$PYTHON_BIN" "$SCRIPT_DIR/align_official_gsk3b_candidates.py" \
  --official-reference "$OFFICIAL_GSK3B" \
  --candidates "$OUTPUT_DIR/c5_table1_n20_candidates.csv" \
  --output-csv "$OUTPUT_DIR/c5_official_gsk3b_n20_candidates.csv"

"$PYTHON_BIN" "$SHARED_PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$OFFICIAL_GSK3B" \
  --candidates "$OUTPUT_DIR/c5_official_gsk3b_n20_candidates.csv" \
  --output-dir "$OUTPUT_DIR/official_gsk3b_metrics_any20" \
  --candidate-limit 20 \
  --model-name joint_graph_fragment_categorical_c5_official_gsk3b \
  --task-filter table1 \
  --missing-oracle-policy fail

"$PYTHON_BIN" "$SCRIPT_DIR/collect_c1_table1_n20.py" \
  --sampling-summary "$OUTPUT_DIR/sampling_summary.json" \
  --metrics-json "$OUTPUT_DIR/moledit_table_metrics_any20/moledit_table_summary.json" \
  --preregistration "$SCRIPT_DIR/joint_graph_fragment_categorical_c5_preregistration.json" \
  --official-gsk3b-json "$OUTPUT_DIR/official_gsk3b_metrics_any20/moledit_table_summary.json" \
  --output-json "$OUTPUT_DIR/summary.json"

echo "summary=$OUTPUT_DIR/summary.json"
