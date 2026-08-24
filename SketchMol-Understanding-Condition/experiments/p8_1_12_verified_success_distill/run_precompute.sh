#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"; cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; module load cuda/12.6 2>/dev/null || true; fi
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; SEED=7; SHARED="$PROJECT_DIR/outputs/p8_1_12_verified_success_distill/shared/seed_${SEED}"; mkdir -p "$SHARED"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/experiments/unified_smiles_generator:$PROJECT_DIR/experiments/p8_1_1_short_transaction:$PROJECT_DIR/experiments/p8_1_9_transaction_outcome_distill${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PYTHON_BIN" "$SCRIPT_DIR/build_verified_pseudopairs.py" \
  --teacher-checkpoint "$PROJECT_DIR/outputs/p8_1_1_short_transaction_r1/seed_${SEED}/policy/umtp_graph_action_policy.pt" \
  --student-checkpoint "$PROJECT_DIR/outputs/p8_1_4_full_smiles_multitask_r1/seed_${SEED}/policy/unified_smiles_generator.pt" \
  --train-csv "$PROJECT_DIR/outputs/umtp_graph_action_instruction_v2/seed_${SEED}/data/action_train_instruction_v2.csv" \
  --train-features-dir "$PROJECT_DIR/outputs/unified_smiles_generator_suite_v1/feature_variants/train_condition_features_hf_vlm" \
  --eval-csv "$PROJECT_DIR/outputs/p6_unified_transition_policy_v1/seed_${SEED}/data/edit_table1_gate.csv" \
  --r1-output "$SHARED/r1_verified_uniform.csv" --r2-output "$SHARED/r2_verified_confidence.csv" --audit-output "$SHARED/coverage_audit.json" \
  --limit "${P8112_TEACHER_ROWS:-768}" --max-actions "${P8112_MAX_ACTIONS:-512}" --seed "$SEED" --device auto
touch "$SHARED/PRECOMPUTE_COMPLETE"
