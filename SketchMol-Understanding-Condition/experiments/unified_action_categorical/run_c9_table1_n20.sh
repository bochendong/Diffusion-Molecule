#!/usr/bin/env bash
# C9: C5 warm-start GRPO with KL to frozen C5. DRD2 duplicated, RB not.

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
OUTPUT_DIR="${SUCC_C9_OUTPUT_DIR:-$SHARED_PROJECT_DIR/outputs/joint_graph_fragment_categorical_c9}"
EVAL_CSV="${SUCC_C9_EVAL_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_joint_v2/dataset/table1_test_rows.csv}"
TRAIN_MOLEDIT="${SUCC_C9_TRAIN_MOLEDIT_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_suite_v1/dataset/table1_train_pack/table1_moledit_rows.csv}"
TRAIN_CONDITION="${SUCC_C9_TRAIN_CONDITION_CSV:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_suite_v1/dataset/table1_train_pack/table1_benchmark_condition_rows.csv}"
TRAIN_CSV="${SUCC_C9_TRAIN_CSV:-$OUTPUT_DIR/dataset/table1_train_real_drd2dup_rows.csv}"
GRAPH_CANDIDATES="${SUCC_C9_GRAPH_CANDIDATES:-$SHARED_PROJECT_DIR/outputs/umtp_graph_action_full_eval_v1/seed_7/eval/action/table1/candidate_pool/graph_action_candidates.csv}"
OFFICIAL_GSK3B="${SUCC_C9_OFFICIAL_GSK3B_CSV:-$SHARED_PROJECT_DIR/outputs/direct_smiles_moledit_table1_gsk3b_n20_pilot_v1/gsk3b_pack/table1_eval_gsk3b_moledit_rows.csv}"
BASE_CHECKPOINT="${SUCC_C9_BASE_CHECKPOINT:-$SHARED_PROJECT_DIR/outputs/joint_graph_fragment_categorical_c5/umtp_graph_action_policy_grpo.pt}"
TRAIN_FEATURES="${SUCC_C9_TRAIN_FEATURES:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_suite_v1/feature_variants/train_condition_features_hf_vlm}"
EVAL_FEATURES="${SUCC_C9_EVAL_FEATURES:-$SHARED_PROJECT_DIR/outputs/unified_smiles_generator_suite_v1/feature_variants/eval_condition_features_hf_vlm}"
B31_DIR="${SUCC_B31_DIR:-$SHARED_PROJECT_DIR/outputs/assay_joint_site_token_latent_v31/seed_1931}"
REPRESENTATION_DIR="${SUCC_ASSAY_JOINT_REPRESENTATION_DIR:-$SHARED_PROJECT_DIR/outputs/graph_latent_autoencoder_v2/seed_1725}"
FRAGMENT_DIR="${SUCC_ASSAY_JOINT_FRAGMENT_DIR:-$SHARED_PROJECT_DIR/outputs/latent_fragment_attachment_kernel_v24/cpu_seed_1761}"
TRAIN_DEVICE="${SUCC_C9_TRAIN_DEVICE:-auto}"
EVAL_DEVICE="${SUCC_DEVICE:-cpu}"
EVAL_LIMIT="${SUCC_C9_EVAL_LIMIT:-0}"
TRAIN_LIMIT="${SUCC_C9_TRAIN_LIMIT:-0}"
SCORE_BATCH="${SUCC_C9_SCORE_BATCH:-64}"

for path in \
  "$EVAL_CSV" \
  "$TRAIN_MOLEDIT" \
  "$TRAIN_CONDITION" \
  "$GRAPH_CANDIDATES" \
  "$OFFICIAL_GSK3B" \
  "$BASE_CHECKPOINT" \
  "$B31_DIR/assay_joint_site_token_energy.pt" \
  "$REPRESENTATION_DIR/graph_latent_autoencoder.pt" \
  "$REPRESENTATION_DIR/summary.json" \
  "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  "$TRAIN_FEATURES/index.csv" \
  "$WORKTREE_LATENT/assay_joint_site_token_latent_v31_preregistration.json" \
  "$SCRIPT_DIR/joint_graph_fragment_categorical_c9_preregistration.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing input: $path" >&2; exit 2; }
done
for path in "$TRAIN_FEATURES" "$EVAL_FEATURES"; do
  [[ -d "$path" ]] || { echo "ERROR: missing feature dir: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR/dataset"
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator:$PROJECT_DIR/scripts:$WORKTREE_LATENT:$WORKTREE_LATENT/../..:$WORKTREE_LATENT/../../experiments/unified_constraint_agent:$REPO_DIR/SketchMol-Unified-3MDiffusion/scripts:$PROJECT_DIR:$REPO_DIR/SketchMol-Unified-3MDiffusion${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$SHARED_PROJECT_DIR/inputs/tdc_oracles/gsk3b_legacy_sklearn_compatible.pkl}"

echo "C9 KL-to-C5 graph GRPO + C1 mixture n=20"
echo "  python=$PYTHON_BIN"
echo "  train_device=$TRAIN_DEVICE"
echo "  score_batch=$SCORE_BATCH"
echo "  output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_c7_real_train.py" \
  --train-moledit-csv "$TRAIN_MOLEDIT" \
  --train-condition-csv "$TRAIN_CONDITION" \
  --feature-index-csv "$TRAIN_FEATURES/index.csv" \
  --output-csv "$TRAIN_CSV" \
  --output-json "$OUTPUT_DIR/dataset/prepare_summary.json" \
  --duplicate-tasks "DRD2:decrease+MW:decrease+SA:decrease"

"$PYTHON_BIN" "$SCRIPT_DIR/train_graph_action_grpo_c5.py" \
  --c5-protocol-manifest "$SCRIPT_DIR/joint_graph_fragment_categorical_c9_preregistration.json" \
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
  --c1-protocol-manifest "$SCRIPT_DIR/joint_graph_fragment_categorical_c9_preregistration.json" \
  --output-dir "$OUTPUT_DIR" \
  --candidate-output "$OUTPUT_DIR/c9_table1_n20_candidates.csv" \
  --device "$EVAL_DEVICE" \
  --eval-limit "$EVAL_LIMIT"

"$PYTHON_BIN" "$SHARED_PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$EVAL_CSV" \
  --candidates "$OUTPUT_DIR/c9_table1_n20_candidates.csv" \
  --output-dir "$OUTPUT_DIR/moledit_table_metrics_any20" \
  --candidate-limit 20 \
  --model-name joint_graph_fragment_categorical_c9 \
  --task-filter table1 \
  --missing-oracle-policy fail

"$PYTHON_BIN" "$SCRIPT_DIR/align_official_gsk3b_candidates.py" \
  --official-reference "$OFFICIAL_GSK3B" \
  --candidates "$OUTPUT_DIR/c9_table1_n20_candidates.csv" \
  --output-csv "$OUTPUT_DIR/c9_official_gsk3b_n20_candidates.csv"

"$PYTHON_BIN" "$SHARED_PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" \
  --reference "$OFFICIAL_GSK3B" \
  --candidates "$OUTPUT_DIR/c9_official_gsk3b_n20_candidates.csv" \
  --output-dir "$OUTPUT_DIR/official_gsk3b_metrics_any20" \
  --candidate-limit 20 \
  --model-name joint_graph_fragment_categorical_c9_official_gsk3b \
  --task-filter table1 \
  --missing-oracle-policy fail

"$PYTHON_BIN" "$SCRIPT_DIR/collect_c1_table1_n20.py" \
  --sampling-summary "$OUTPUT_DIR/sampling_summary.json" \
  --metrics-json "$OUTPUT_DIR/moledit_table_metrics_any20/moledit_table_summary.json" \
  --preregistration "$SCRIPT_DIR/joint_graph_fragment_categorical_c9_preregistration.json" \
  --official-gsk3b-json "$OUTPUT_DIR/official_gsk3b_metrics_any20/moledit_table_summary.json" \
  --output-json "$OUTPUT_DIR/summary.json"

echo "summary=$OUTPUT_DIR/summary.json"
