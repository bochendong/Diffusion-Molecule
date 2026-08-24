#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="$(cd "$HERE/../../.." && pwd)"; REPO="$(cd "$PROJECT/.." && pwd)"; cd "$REPO"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; module load cuda/12.6 2>/dev/null || true; fi
PY="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; SEED=7; P811="$PROJECT/outputs/p8_1_1_short_transaction_r1/seed_${SEED}"; P6="$PROJECT/outputs/p6_unified_transition_policy_v1/seed_${SEED}"; DIRECT="$PROJECT/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"; ROOT="$PROJECT/outputs/p8_2_state_adaptive_molprogram/transaction_group_rl"; OUT="$ROOT/base_replay/seed_${SEED}/eval/denovo"; mkdir -p "$OUT"
export PYTHONPATH="$PROJECT:$PROJECT/experiments/unified_smiles_generator:$PROJECT/experiments/p8_1_1_short_transaction${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TOKENIZERS_PARALLELISM=false
"$PY" "$PROJECT/experiments/p8_1_1_short_transaction/sample_raw_denovo.py" --checkpoint "$P811/policy/umtp_graph_action_policy.pt" --eval-csv "$P6/data/denovo_hard_gate.csv" --eval-features-dir "$DIRECT/eval_condition_features_hf_vlm" --output-csv "$OUT/candidates.csv" --summary-json "$OUT/sampling_summary.json" --num-samples 20 --seed 1982 --device auto
"$PY" "$PROJECT/experiments/p6_unified_molecular_transition_policy/evaluate_p6_denovo_gate.py" --eval-csv "$P6/data/denovo_hard_gate.csv" --candidates-csv "$OUT/candidates.csv" --output-json "$OUT/metrics.json" --output-md "$OUT/report.md" --budgets 1,8,20
"$PY" "$HERE/compare_denovo_replay.py" --base-dir "$OUT" --r1-dir "$ROOT/r1/seed_${SEED}/eval/denovo" --r2-dir "$ROOT/r2/seed_${SEED}/eval/denovo" --output "$ROOT/base_replay/seed_${SEED}/functional_protection_audit.json"
touch "$ROOT/base_replay/seed_${SEED}/COMPLETE"
