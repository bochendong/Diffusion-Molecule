#!/usr/bin/env bash
# One stage of the CPU-only, train-only MuMO v8 parallel evidence pipeline.

set -euo pipefail

STAGE="${1:?usage: run_mumo_parallel_evidence_v8.sh prepare|delta|features|verifier|finalize}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CODE_PROJECT_DIR="$REPO_DIR/SketchMol-Understanding-Condition"

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load StdEnv/2023
  module load python/3.11
  module load rdkit/2025.09.4
fi

PYTHON_BIN="${SUCC_UCA_PYTHON_BIN:-/home/bdong/.venvs/molscribe_overlay/bin/python}"
SHARED_REPO_DIR="${SUCC_UCA_SHARED_REPO_DIR:-/scratch/bdong/projects/Diffusion-Molecule}"
PROJECT_DIR="$SHARED_REPO_DIR/SketchMol-Understanding-Condition"
RUN_ROOT="${SUCC_UCA_MUMO_PARALLEL_ROOT:-$PROJECT_DIR/outputs/unified_constraint_agent_mumo_parallel_evidence_v8/seed_1711}"
MUMO_TRAIN="${SUCC_UCA_MUMO_TRAIN_JSON:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/train.json}"
MUMO_TEST="${SUCC_UCA_MUMO_TEST_JSON:-/scratch/bdong/datasets/Diffusion-Molecule/external/mumo/test.json}"
AUDIT_CSV="${SUCC_UCA_AUDIT_CSV:-$PROJECT_DIR/outputs/unified_constraint_agent_hierarchical_support_v4/data/support_audit_disjoint_rows.csv}"
SHARD_COUNT="${SUCC_UCA_MUMO_SHARDS:-32}"
SEED="${SUCC_UCA_SEED:-1711}"

export PYTHONPATH="$CODE_PROJECT_DIR:$REPO_DIR/SketchMol-MultiProperty-EditDataset${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$RUN_ROOT/data" "$RUN_ROOT/delta" "$RUN_ROOT/features" "$RUN_ROOT/verifiers" "$RUN_ROOT/merged"

case "$STAGE" in
  prepare)
    for path in "$PYTHON_BIN" "$MUMO_TRAIN" "$MUMO_TEST" "$AUDIT_CSV"; do
      [[ -e "$path" ]] || { echo "ERROR: missing v8 input: $path" >&2; exit 2; }
    done
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_mumo_parallel_train.py" \
      --train-json "$MUMO_TRAIN" \
      --test-json-digest-only "$MUMO_TEST" \
      --audit-csv "$AUDIT_CSV" \
      --output-dir "$RUN_ROOT/data" \
      --rows-per-task "${SUCC_UCA_ROWS_PER_TASK:-5500}" \
      --dev-fraction "${SUCC_UCA_DEV_FRACTION:-0.10}" \
      --shard-count "$SHARD_COUNT" \
      --seed "$SEED"
    ;;
  delta)
    INDEX="${SLURM_ARRAY_TASK_ID:?delta stage requires SLURM_ARRAY_TASK_ID}"
    PADDED="$(printf '%03d' "$INDEX")"
    "$PYTHON_BIN" "$SCRIPT_DIR/build_mumo_delta_shard.py" \
      --input-jsonl "$RUN_ROOT/data/train_shard_${PADDED}.jsonl" \
      --output-jsonl "$RUN_ROOT/delta/transforms_${PADDED}.jsonl" \
      --manifest-json "$RUN_ROOT/delta/manifest_${PADDED}.json"
    ;;
  features)
    INDEX="${SLURM_ARRAY_TASK_ID:?features stage requires SLURM_ARRAY_TASK_ID}"
    PADDED="$(printf '%03d' "$INDEX")"
    "$PYTHON_BIN" "$SCRIPT_DIR/build_mumo_feature_shard.py" \
      --input-jsonl "$RUN_ROOT/data/train_shard_${PADDED}.jsonl" \
      --output-npz "$RUN_ROOT/features/features_${PADDED}.npz" \
      --manifest-json "$RUN_ROOT/features/manifest_${PADDED}.json" \
      --fingerprint-bits 2048 \
      --fingerprint-radius 2
    ;;
  verifier)
    INDEX="${SLURM_ARRAY_TASK_ID:?verifier stage requires SLURM_ARRAY_TASK_ID}"
    PROPERTIES=(bbbp drd2 hia mutagenicity plogp)
    PROPERTY="${PROPERTIES[$INDEX]:?invalid verifier array index $INDEX}"
    "$PYTHON_BIN" "$SCRIPT_DIR/train_mumo_property_verifier.py" \
      --feature-dir "$RUN_ROOT/features" \
      --property "$PROPERTY" \
      --output-model "$RUN_ROOT/verifiers/$PROPERTY/model.joblib" \
      --metrics-json "$RUN_ROOT/verifiers/$PROPERTY/metrics.json" \
      --seed "$SEED" \
      --estimators "${SUCC_UCA_VERIFIER_TREES:-96}" \
      --max-fit-molecules "${SUCC_UCA_MAX_FIT_MOLECULES:-100000}" \
      --jobs "${SLURM_CPUS_PER_TASK:-8}"
    ;;
  finalize)
    "$PYTHON_BIN" "$SCRIPT_DIR/finalize_mumo_parallel_evidence.py" \
      --run-root "$RUN_ROOT" \
      --shard-count "$SHARD_COUNT" \
      --min-label-coverage 0.95 \
      --min-threshold-recall 0.85 \
      --min-dev-pairs 100 \
      --min-unique-transforms 10000
    ;;
  *)
    echo "ERROR: unknown v8 stage: $STAGE" >&2
    exit 2
    ;;
esac
