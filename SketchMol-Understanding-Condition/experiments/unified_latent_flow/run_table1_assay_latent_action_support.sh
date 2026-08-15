#!/usr/bin/env bash
# B30-r1: one fail-closed assay-support shard over the exhaustive B24 vocabulary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHARED_REPO_DIR="${SUCC_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
SHARED_PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
PYTHON_BIN="${SUCC_ASSAY_SUPPORT_PYTHON:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
OUTPUT_ROOT="${SUCC_ASSAY_SUPPORT_OUTPUT_ROOT:-$SHARED_PROJECT_DIR/outputs/table1_assay_latent_action_support_v30_r1/seed_1922}"
FRAGMENT_DIR="${SUCC_ASSAY_SUPPORT_FRAGMENT_DIR:-$SHARED_PROJECT_DIR/outputs/latent_fragment_attachment_kernel_v24/cpu_seed_1761}"
B29_DIR="${SUCC_ASSAY_SUPPORT_B29_DIR:-$SHARED_PROJECT_DIR/outputs/table1_energy_tilted_latent_transfer_v29/seed_1911}"
PREREGISTRATION="$SCRIPT_DIR/table1_assay_latent_action_support_v30_r1_preregistration.json"
ORACLE_DIR="${SUCC_ASSAY_ORACLE_DIR:-$SHARED_PROJECT_DIR/inputs/tdc_oracles}"
GSK3B_ORACLE="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
DRD2_ORACLE="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?B30 requires SLURM_ARRAY_TASK_ID}"
SHARD_DIR="$(printf '%s/shards/shard_%03d' "$OUTPUT_ROOT" "$SHARD_INDEX")"

for path in \
  "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  "$B29_DIR/summary.json" \
  "$B29_DIR/nearest_token_candidates.csv" \
  "$GSK3B_ORACLE" \
  "$DRD2_ORACLE" \
  "$PREREGISTRATION"; do
  [[ -f "$path" ]] || { echo "ERROR: missing B30 input: $path" >&2; exit 2; }
done

if [[ -f "$SHARD_DIR/summary.json" ]]; then
  echo "ERROR: completed B30 shard exists: $SHARD_DIR/summary.json" >&2
  exit 2
fi

mkdir -p "$SHARD_DIR"
export PYTHONPATH="$PROJECT_DIR:$SHARED_PROJECT_DIR:$PROJECT_DIR/experiments/unified_constraint_agent${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON_BIN" "$SCRIPT_DIR/audit_table1_assay_latent_action_support.py" \
  --fragment-checkpoint "$FRAGMENT_DIR/latent_fragment_attachment_kernel.pt" \
  --b29-summary "$B29_DIR/summary.json" \
  --b29-candidates "$B29_DIR/nearest_token_candidates.csv" \
  --protocol-manifest "$PREREGISTRATION" \
  --gsk3b-oracle "$GSK3B_ORACLE" \
  --drd2-oracle "$DRD2_ORACLE" \
  --output-dir "$SHARD_DIR" \
  --shard-index "$SHARD_INDEX"
