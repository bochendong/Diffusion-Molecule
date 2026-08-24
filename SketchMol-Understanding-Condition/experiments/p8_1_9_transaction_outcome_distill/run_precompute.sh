#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"; REPO_DIR="$(cd "$PROJECT_DIR/.." && pwd)"; cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; module load cuda/12.6 2>/dev/null || true; fi
PYTHON_BIN="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; SEED=7
PROJECT="$REPO_DIR/SketchMol-Understanding-Condition"; SHARED="$PROJECT/outputs/p8_1_9_transaction_outcome_distill/shared/seed_${SEED}"; mkdir -p "$SHARED"
export PYTHONPATH="$PROJECT:$PROJECT/experiments/unified_smiles_generator:$PROJECT/experiments/p8_1_1_short_transaction${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PYTHON_BIN" "$SCRIPT_DIR/build_teacher_pseudopairs.py" \
  --teacher-checkpoint "$PROJECT/outputs/p8_1_1_short_transaction_r1/seed_${SEED}/policy/umtp_graph_action_policy.pt" \
  --student-checkpoint "$PROJECT/outputs/p8_1_4_full_smiles_multitask_r1/seed_${SEED}/policy/unified_smiles_generator.pt" \
  --train-csv "$PROJECT/outputs/umtp_graph_action_instruction_v2/seed_${SEED}/data/action_train_instruction_v2.csv" \
  --train-features-dir "$PROJECT/outputs/unified_smiles_generator_suite_v1/feature_variants/train_condition_features_hf_vlm" \
  --eval-csv "$PROJECT/outputs/p6_unified_transition_policy_v1/seed_${SEED}/data/edit_table1_gate.csv" \
  --r1-output "$SHARED/r1_uniform.csv" --r2-output "$SHARED/r2_confidence_weighted.csv" --audit-output "$SHARED/data_audit.json" \
  --limit "${P819_TEACHER_ROWS:-768}" --seed "$SEED" --device auto
touch "$SHARED/PRECOMPUTE_COMPLETE"
