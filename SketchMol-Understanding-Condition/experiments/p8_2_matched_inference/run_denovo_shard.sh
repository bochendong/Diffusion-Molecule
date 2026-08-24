#!/usr/bin/env bash
set -euo pipefail
PC="${1:?property count 2--7}"; [[ "$PC" =~ ^[2-7]$ ]] || exit 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PROJECT="$(cd "$HERE/../.." && pwd)"; REPO="$(cd "$PROJECT/.." && pwd)"; cd "$REPO"
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1 || true; module load StdEnv/2023 python/3.11 rdkit/2025.09.4; module load cuda/12.6 2>/dev/null || true; fi
PY="${SUCC_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"; OUT="$PROJECT/outputs/p8_2_matched_inference/seed_7"; CHECKPOINT="$PROJECT/outputs/p8_1_1_short_transaction_r1/seed_7/policy/umtp_graph_action_policy.pt"; DIRECT="$PROJECT/outputs/direct_smiles_denovo_2p7p_v2_mixed_condition"; SHARD="$OUT/denovo/pc$PC"
export PYTHONPATH="$PROJECT:$PROJECT/experiments/unified_smiles_generator:$PROJECT/scripts${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TOKENIZERS_PARALLELISM=false
mkdir -p "$SHARD"
"$PY" "$PROJECT/experiments/p8_1_1_short_transaction/sample_raw_denovo.py" --checkpoint "$CHECKPOINT" --eval-csv "$OUT/data/pc${PC}_inference.csv" --eval-features-dir "$DIRECT/eval_condition_features_hf_vlm" --output-csv "$SHARD/candidates.csv" --summary-json "$SHARD/sampling_summary.json" --num-samples 20 --seed "$((8200 + PC))" --device auto
touch "$SHARD/COMPLETE"
