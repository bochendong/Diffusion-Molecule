#!/bin/bash
set -euo pipefail

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

cat > "$TMPDIR/smiles.csv" <<'CSV'
smiles
c1ccccc1
Fc1ccccc1
Clc1ccccc1
Brc1ccccc1
Cc1ccccc1
COc1ccccc1
Oc1ccccc1
Nc1ccccc1
CCc1ccccc1
CCOc1ccccc1
CC(=O)Nc1ccccc1
CC(=O)Oc1ccccc1
CCN(CC)CC
CCCN(CC)CC
CCO
CCCO
CCCCO
CCCl
CCCCl
CCBr
CSV

python3 -m phystabmol.instruction_dataset \
  --data "$TMPDIR/smiles.csv" \
  --out "$TMPDIR/instruction.csv" \
  --limit 50 \
  --max-pairs 30 \
  --pairs-per-source 10 \
  --instructions-per-spec 2 \
  --reference-pool-size 8 \
  --min-similarity 0.05 \
  --max-similarity 0.99 \
  --seed 5

python3 -m phystabmol.instruction_baselines \
  --dataset "$TMPDIR/instruction.csv" \
  --baseline oracle_target \
  --split all \
  --out "$TMPDIR/oracle.csv"

python3 -m phystabmol.instruction_evaluate \
  --candidates "$TMPDIR/oracle.csv" \
  --train-dataset "$TMPDIR/instruction.csv" \
  --out "$TMPDIR/oracle_metrics.json"

python3 -m phystabmol.instruction_experiment \
  --dataset "$TMPDIR/instruction.csv" \
  --output-dir "$TMPDIR/runs" \
  --run-name smoke_instruction \
  --backend sklearn \
  --eval-limit 8 \
  --samples-per-instruction 2 \
  --decode-top-k 1 \
  --multimodal-context source_reference \
  --timesteps 8 \
  --noise-repeats 2 \
  --sklearn-hidden 32 32

echo "Instruction editing smoke test passed."
