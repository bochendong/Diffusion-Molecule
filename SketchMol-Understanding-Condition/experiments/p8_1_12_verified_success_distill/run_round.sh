#!/usr/bin/env bash
set -euo pipefail
ROUND="${1:?r1 or r2}"; [[ "$ROUND" == r1 || "$ROUND" == r2 ]] || exit 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"; cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; module load cuda/12.6 2>/dev/null || true; fi
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; SEED=7
P814="$PROJECT_DIR/outputs/p8_1_4_full_smiles_multitask_r1/seed_${SEED}"; P6="$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}"; DIRECT="$PROJECT_DIR/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"; JOINT="$PROJECT_DIR/outputs/unified_smiles_generator_joint_v2"
BASE="$P814/policy/unified_smiles_generator.pt"; SHARED="$PROJECT_DIR/outputs/p8_1_12_verified_success_distill/shared/seed_${SEED}"; OUT="$PROJECT_DIR/outputs/p8_1_12_verified_success_distill/${ROUND}/seed_${SEED}"; mkdir -p "$OUT/policy" "$OUT/eval/denovo/raw" "$OUT/eval/edit/raw"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator:$PROJECT_DIR/experiments/p8_1_1_short_transaction:$PROJECT_DIR/experiments/p8_1_9_transaction_outcome_distill${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [[ ! -s "$SHARED/r1_verified_uniform.csv" || ! -s "$SHARED/r2_verified_confidence.csv" || ! -s "$SHARED/coverage_audit.json" ]]; then bash "$SCRIPT_DIR/run_precompute.sh"; fi
TRAIN="$SHARED/r1_verified_uniform.csv"; [[ "$ROUND" == r2 ]] && TRAIN="$SHARED/r2_verified_confidence.csv"
"$PYTHON_BIN" "$PROJECT_DIR/experiments/p8_1_7_source_clamped_policy/source_clamped_entrypoint.py" train \
  --resume-checkpoint "$BASE" --reset-training-state --train-csv "$TRAIN" --condition-features-dir "$PROJECT_DIR/outputs/unified_smiles_generator_suite_v1/feature_variants/train_condition_features_hf_vlm" \
  --condition-layout unified --output-dir "$OUT/policy" --trainable-scope source_only --max-smiles-length 160 --epochs "${P8112_EPOCHS:-4}" --batch-size 64 \
  --sampling-mode task_balanced --samples-per-epoch "${P8112_SAMPLES_PER_EPOCH:-3072}" --lr "${P8112_LR:-8e-5}" --weight-decay 0 --seed "$SEED" --device auto
STUDENT="$OUT/policy/unified_smiles_generator.pt"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_student.py" --base "$BASE" --student "$STUDENT" --data-audit "$SHARED/coverage_audit.json" --round "$ROUND" --output "$OUT/pre_eval_audit.json"
sample(){ local mode="$1" csv="$2" features="$3" seed="$4"; "$PYTHON_BIN" "$PROJECT_DIR/experiments/p8_1_7_source_clamped_policy/source_clamped_entrypoint.py" sample --checkpoint "$STUDENT" --eval-csv "$csv" --eval-condition-features-dir "$features" --condition-layout unified --output-dir "$OUT/eval/$mode/raw" --prediction-csv "$OUT/eval/$mode/selected.csv" --candidate-output-csv "$OUT/eval/$mode/candidates.csv" --method "p8_1_12_${ROUND}" --decoding-mode sample --num-samples 20 --top-k-candidates 20 --max-candidates 20 --disable-finalizer --smiles-grammar-constraint --max-new-tokens 120 --temperature 0.80 --top-k 32 --top-p 0.95 --parallel-samples 10 --max-parallel-sequences 256 --seed "$seed" --device auto; }
sample denovo "$P6/data/denovo_hard_gate.csv" "$DIRECT/eval_condition_features_hf_vlm" 1812
sample edit "$P6/data/edit_table1_gate.csv" "$JOINT/feature_variants/validation_condition_features_hf_vlm" 2812
"$PYTHON_BIN" "$PROJECT_DIR/experiments/p8_1_7_source_clamped_policy/normalize_denovo_candidates.py" --input "$OUT/eval/denovo/candidates.csv" --output "$OUT/eval/denovo/candidates_normalized.csv"
"$PYTHON_BIN" "$PROJECT_DIR/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" --eval-csv "$P6/data/denovo_hard_gate.csv" --candidates-csv "$OUT/eval/denovo/candidates_normalized.csv" --output-json "$OUT/eval/denovo/metrics.json" --output-md "$OUT/eval/denovo/report.md" --budgets 1,8,20
for budget in 1 8 20; do "$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" --reference "$P6/data/edit_table1_gate.csv" --candidates "$OUT/eval/edit/candidates.csv" --output-dir "$OUT/eval/edit/any${budget}" --candidate-limit "$budget" --model-name "p8_1_12_${ROUND}_any${budget}" --task-filter table1 --missing-oracle-policy fail; done
"$PYTHON_BIN" "$PROJECT_DIR/scripts/evaluate_moledit_table1_anyk.py" --reference "$P6/data/edit_table1_gate.csv" --candidates "$OUT/eval/edit/candidates.csv" --output-dir "$OUT/eval/edit/candidate20" --candidate-limit 20 --aggregation candidate --model-name "p8_1_12_${ROUND}_candidate20" --task-filter table1 --missing-oracle-policy fail
"$PYTHON_BIN" "$PROJECT_DIR/experiments/p8_1_7_source_clamped_policy/audit_candidates.py" --checkpoint "$STUDENT" --candidates "$OUT/eval/edit/candidates.csv" --scale 1.0 --output "$OUT/source_audit.json"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_student.py" --base "$BASE" --student "$STUDENT" --data-audit "$SHARED/coverage_audit.json" --round "$ROUND" --output "$OUT/final_audit.json"
touch "$OUT/COMPLETE"; echo "P8.1.12 $ROUND complete: $OUT"
