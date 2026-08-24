#!/usr/bin/env bash
set -euo pipefail
ROUND="${1:?r1 or r2}"; [[ "$ROUND" == r1 || "$ROUND" == r2 ]] || exit 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"; cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; module load cuda/12.6 2>/dev/null || true; fi
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; SEED=7
P812="$PROJECT_DIR/experiments/p8_1_2_unified_transduction"; P6="$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}"; DIRECT="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"; SUITE="$PROJECT_DIR/outputs/unified_smiles_generator_suite_v1"; JOINT="$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2"
ORACLE="$PROJECT_DIR/outputs/p8_1_2_unified_transduction_v1/seed_${SEED}"; BASE_ROOT="$PROJECT_DIR/outputs/p8_1_2_unified_transduction_raw_v1/seed_${SEED}"; BASE="$BASE_ROOT/policy/umtp_graph_action_policy.pt"
OUT="$PROJECT_DIR/outputs/p8_1_11_transduction_group_rl/${ROUND}/seed_${SEED}"; AGG=joint_bottleneck; [[ "$ROUND" == r2 ]] && AGG=dense_softmin
mkdir -p "$OUT/data" "$OUT/policy" "$OUT/eval/denovo" "$OUT/eval/edit"; export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator:$P812${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PYTHON_BIN" "$SCRIPT_DIR/preflight.py" --checkpoint "$BASE" --training-summary "$BASE_ROOT/policy/training_summary.json" --output "$OUT/preflight.json"
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_train_support.py" --denovo-csv "$DIRECT/denovo_2p7p_train_rows.csv" --edit-csv "$ORACLE/r2/transduction_rows.csv" \
  --denovo-features "$DIRECT/train_condition_features_hf_vlm" --edit-features "$SUITE/feature_variants/train_condition_features_hf_vlm" --output-dir "$OUT/data" --per-denovo-count 32 --edit-limit 64 --seed "$SEED"
"$PYTHON_BIN" "$SCRIPT_DIR/group_relative_reinforce.py" --base-checkpoint "$BASE" --train-csv "$OUT/data/train_rows.csv" --features-dir "$OUT/data/features" \
  --output-dir "$OUT/policy" --reward-aggregation "$AGG" --rollouts 4 --max-prompts 128 --lr 2e-6 --temperature 0.8 --reference-logratio-weight 0.05 --sft-weight 0.10 --seed "$SEED" --device auto
CHECKPOINT="$OUT/policy/transduction_group_relative_reinforce.pt"
"$PYTHON_BIN" "$P812/sample_transduction_policy.py" --checkpoint "$CHECKPOINT" --train-csv "$OUT/data/train_rows.csv" --eval-csv "$P6/data/denovo_hard_gate.csv" \
  --eval-features-dir "$DIRECT/eval_condition_features_hf_vlm" --candidate-output-csv "$OUT/eval/denovo/candidates.csv" --summary-json "$OUT/eval/denovo/sampling_summary.json" --num-samples 20 --max-new-tokens 128 --temperature 0.8 --top-k 32 --top-p 0.95 --seed 1811 --device auto
"$PYTHON_BIN" "$P812/sample_transduction_policy.py" --checkpoint "$CHECKPOINT" --train-csv "$OUT/data/train_rows.csv" --eval-csv "$P6/data/edit_table1_gate.csv" \
  --eval-features-dir "$JOINT/feature_variants/validation_condition_features_hf_vlm" --candidate-output-csv "$OUT/eval/edit/candidates.csv" --summary-json "$OUT/eval/edit/sampling_summary.json" --num-samples 20 --max-new-tokens 64 --temperature 0.8 --top-k 32 --top-p 0.95 --seed 2811 --device auto
"$PYTHON_BIN" "$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" --eval-csv "$P6/data/denovo_hard_gate.csv" --candidates-csv "$OUT/eval/denovo/candidates.csv" --output-json "$OUT/eval/denovo/metrics.json" --output-md "$OUT/eval/denovo/report.md" --budgets 1,8,20
for budget in 1 8 20; do "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" --reference "$P6/data/edit_table1_gate.csv" --candidates "$OUT/eval/edit/candidates.csv" --output-dir "$OUT/eval/edit/any${budget}" --candidate-limit "$budget" --model-name "p8_1_11_${ROUND}_any${budget}" --task-filter table1 --missing-oracle-policy fail; done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" --reference "$P6/data/edit_table1_gate.csv" --candidates "$OUT/eval/edit/candidates.csv" --output-dir "$OUT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate --model-name "p8_1_11_${ROUND}_candidate20" --task-filter table1 --missing-oracle-policy fail
"$PYTHON_BIN" "$SCRIPT_DIR/audit_candidates.py" --candidates "$OUT/eval/edit/candidates.csv" --output "$OUT/candidate_audit.json"; touch "$OUT/COMPLETE"
