#!/bin/bash
set -euo pipefail

# Generate and evaluate deterministic instruction-editing baselines.
#
# Usage:
#   cd PhysTabMol
#   bash scripts/evaluate_instruction_baselines.sh

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"

DATASET="${PHYSTABMOL_INSTRUCTION_DATASET:-data/instruction_editing.csv}"
OUT_DIR="${PHYSTABMOL_BASELINE_OUT_DIR:-outputs/instruction_baselines}"
SPLIT="${PHYSTABMOL_SPLIT:-test}"
LIMIT="${PHYSTABMOL_LIMIT:-0}"
RETRIEVAL_POOL_SIZE="${PHYSTABMOL_RETRIEVAL_POOL_SIZE:-512}"
mkdir -p "$OUT_DIR"

if [[ ! -s "$DATASET" ]]; then
  echo "Instruction dataset not found at $DATASET; build it first:" >&2
  echo "  bash scripts/build_instruction_dataset.sh" >&2
  exit 2
fi

LIMIT_ARGS=()
if [[ "$LIMIT" != "0" ]]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi

for baseline in no_edit random_target rule_retrieval oracle_target; do
  candidates="$OUT_DIR/${baseline}_${SPLIT}_candidates.csv"
  metrics="$OUT_DIR/${baseline}_${SPLIT}_metrics.json"
  details="$OUT_DIR/${baseline}_${SPLIT}_verified.csv"
  python3 -m phystabmol.instruction_baselines \
    --dataset "$DATASET" \
    --baseline "$baseline" \
    --split "$SPLIT" \
    --retrieval-pool-size "$RETRIEVAL_POOL_SIZE" \
    --out "$candidates" \
    "${LIMIT_ARGS[@]}"
  python3 -m phystabmol.instruction_evaluate \
    --candidates "$candidates" \
    --train-dataset "$DATASET" \
    --out "$metrics" \
    --details-out "$details"
done

echo "Baseline results written under $OUT_DIR"
