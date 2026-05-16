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
Cc1ccccc1
COc1ccccc1
Oc1ccccc1
Nc1ccccc1
CCc1ccccc1
CCOc1ccccc1
CSV

python3 -m phystabmol.instruction_dataset \
  --data "$TMPDIR/smiles.csv" \
  --out "$TMPDIR/instruction.csv" \
  --limit 20 \
  --max-pairs 8 \
  --pairs-per-source 8 \
  --instructions-per-spec 2 \
  --min-similarity 0.05 \
  --max-similarity 0.99 \
  --split-strategy scaffold \
  --seed 3

python3 - <<'PY' "$TMPDIR/instruction.csv"
import pandas as pd
import sys

from phystabmol.instruction_actions import ground_instruction_text
from phystabmol.instruction_verifier import verify_instruction

df = pd.read_csv(sys.argv[1])
required = {
    "difficulty",
    "split_by_scaffold",
    "split_random",
    "edit_combo_split",
    "paraphrase_split",
    "instruction_combo_key",
    "threshold_range_key",
}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"missing instruction dataset columns: {sorted(missing)}")
if not {"easy", "medium", "hard"} & set(df["difficulty"]):
    raise SystemExit("difficulty labels were not generated")
row = df.iloc[0]
result = verify_instruction(row["source_smiles"], row["target_smiles"], row["instruction_spec_json"])
if not result["overall_success"]:
    raise SystemExit("target molecule does not satisfy its own instruction spec")
grounded = ground_instruction_text(row["instruction_text"], row["instruction_spec_json"])
if not grounded["consistent_with_base"]:
    raise SystemExit("template instruction failed action grammar consistency")
print("instruction component checks passed")
PY
