#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="$(cd "$HERE/../.." && pwd)"; REPO="$(cd "$PROJECT/.." && pwd)"; cd "$REPO"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; module load cuda/12.6 2>/dev/null || true; fi
PY="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; SEED=7
UP="$PROJECT/outputs/p8_1_12_verified_success_distill/shared/seed_${SEED}"; OUT="$PROJECT/outputs/p8_1_13_verified_counterfactual_dpo/shared/seed_${SEED}"; mkdir -p "$OUT"
[[ -e "$UP/PRECOMPUTE_COMPLETE" && -s "$UP/r1_verified_uniform.csv" && -s "$UP/coverage_audit.json" ]] || { echo "P8.1.12 PRE artifact missing; fail closed" >&2; exit 42; }
export PYTHONPATH="$PROJECT:$PROJECT/experiments/unified_smiles_generator:$PROJECT/experiments/p8_1_1_short_transaction:$PROJECT/experiments/p8_1_9_transaction_outcome_distill${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PY" "$HERE/build_preference_pairs.py" --p8112-positive-csv "$UP/r1_verified_uniform.csv" --p8112-audit "$UP/coverage_audit.json" \
 --teacher-checkpoint "$PROJECT/outputs/p8_1_1_short_transaction_r1/seed_${SEED}/policy/umtp_graph_action_policy.pt" \
 --student-checkpoint "$PROJECT/outputs/p8_1_4_full_smiles_multitask_r1/seed_${SEED}/policy/unified_smiles_generator.pt" \
 --features-dir "$PROJECT/outputs/unified_smiles_generator_suite_v1/feature_variants/train_condition_features_hf_vlm" \
 --eval-csv "$PROJECT/outputs/p6_unified_transition_policy_v1/seed_${SEED}/data/edit_table1_gate.csv" \
 --output-csv "$OUT/preference_pairs.csv" --audit-output "$OUT/pair_audit.json" --max-actions "${P8113_MAX_ACTIONS:-512}" --device auto
touch "$OUT/PRECOMPUTE_COMPLETE"
