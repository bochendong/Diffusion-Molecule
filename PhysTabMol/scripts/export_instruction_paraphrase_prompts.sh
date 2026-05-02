#!/bin/bash
set -euo pipefail

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"

DATASET="${PHYSTABMOL_INSTRUCTION_DATASET:-data/instruction_editing.csv}"
OUT="${PHYSTABMOL_PARAPHRASE_PROMPTS:-data/instruction_paraphrase_prompts.jsonl}"
N="${PHYSTABMOL_PARAPHRASES_PER_ITEM:-8}"

python3 -m phystabmol.instruction_paraphrases export-prompts \
  --dataset "$DATASET" \
  --out "$OUT" \
  --paraphrases-per-item "$N"

echo "Prompt JSONL written to $OUT"
