#!/bin/bash
set -euo pipefail

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"
export PHYSTABMOL_DISABLE_SKLEARN="${PHYSTABMOL_DISABLE_SKLEARN:-1}"
export PHYSTABMOL_DISABLE_SKLEARN_NN="${PHYSTABMOL_DISABLE_SKLEARN_NN:-1}"

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

if python3 - <<'PY'
from phystabmol.diffusion import SKLEARN_AVAILABLE
raise SystemExit(0 if SKLEARN_AVAILABLE else 1)
PY
then
  python3 -m phystabmol.instruction_experiment \
    --dataset "$TMPDIR/instruction.csv" \
    --output-dir "$TMPDIR/runs" \
    --run-name smoke_instruction \
    --backend sklearn \
    --eval-limit 8 \
    --samples-per-instruction 2 \
    --decode-top-k 1 \
    --multimodal-context source_reference \
    --mmp-pool-size 48 \
    --mmp-verify-candidates 32 \
    --source-aware-pool-size 32 \
    --source-aware-verify-candidates 24 \
    --timesteps 8 \
    --noise-repeats 2 \
    --sklearn-hidden 32 32
else
  echo "scikit-learn is unavailable; running source-aware decoder smoke without diffusion training."
  python3 - <<'PY' "$TMPDIR/instruction.csv"
import pandas as pd
import sys

from phystabmol.features import table_row_from_smiles
from phystabmol.instruction_mmp_decoder import MMPDecoderConfig, MMPTransformationIndex, decode_mmp_transform
from phystabmol.instruction_source_decoder import SourceAwareCandidateIndex, SourceAwareDecoderConfig, decode_source_aware
from phystabmol.instruction_verifier import verify_instruction

df = pd.read_csv(sys.argv[1])
train = df[df["split"] == "train"] if "split" in df else df
if train.empty:
    train = df
mmp_index = MMPTransformationIndex.from_dataframe(train)
index = SourceAwareCandidateIndex.from_dataframe(train)
row = df.iloc[0]
plan = table_row_from_smiles(row["target_smiles"])
mmp_candidates = decode_mmp_transform(
    plan,
    row,
    mmp_index,
    top_k=3,
    config=MMPDecoderConfig(pool_size=48, verify_candidates=32),
)
if not mmp_candidates:
    raise SystemExit("MMP decoder returned no candidates")
source_candidates = decode_source_aware(
    plan,
    row,
    index,
    top_k=3,
    config=SourceAwareDecoderConfig(pool_size=32, verify_candidates=24),
)
if not source_candidates:
    raise SystemExit("source-aware decoder returned no candidates")
result = verify_instruction(row["source_smiles"], source_candidates[0].smiles, row["instruction_spec_json"])
mmp_result = verify_instruction(row["source_smiles"], mmp_candidates[0].smiles, row["instruction_spec_json"])
print("mmp_top_candidate=", mmp_candidates[0].smiles, "valid=", mmp_result["valid"])
print("source_top_candidate=", source_candidates[0].smiles, "constraint_success=", result["constraint_success"])
if not result["constraint_success"]:
    raise SystemExit("source-aware decoder failed the source-preserving constraint smoke")
PY
fi

echo "Instruction editing smoke test passed."
