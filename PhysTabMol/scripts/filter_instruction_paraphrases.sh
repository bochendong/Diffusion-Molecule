#!/bin/bash
set -euo pipefail

# Filter external LLM paraphrases with the deterministic instruction grammar.
#
# Expected input columns/fields:
#   paraphrase_base_id, pair_id, instruction_text
#
# Usage:
#   PHYSTABMOL_LLM_PARAPHRASES=llm_paraphrases.jsonl bash scripts/filter_instruction_paraphrases.sh

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"
if command -v module >/dev/null 2>&1 && [[ "${PHYSTABMOL_USE_MODULE_ENV:-1}" == "1" && -f "$PHYSTABMOL_ROOT/scripts/env_module_venv.sh" ]]; then
  # shellcheck source=/dev/null
  source "$PHYSTABMOL_ROOT/scripts/env_module_venv.sh"
else
  # shellcheck source=/dev/null
  source "$PHYSTABMOL_ROOT/scripts/ensure_phystabmol_venv.sh"
fi

DATASET="${PHYSTABMOL_INSTRUCTION_DATASET:-data/instruction_editing.csv}"
PARAPHRASES="${PHYSTABMOL_LLM_PARAPHRASES:?Set PHYSTABMOL_LLM_PARAPHRASES to a CSV/JSONL returned by your LLM paraphrase workflow.}"
OUT="${PHYSTABMOL_LLM_VERIFIED_OUT:-data/instruction_editing_llm_verified.csv}"
REJECTED="${PHYSTABMOL_LLM_REJECTED_OUT:-data/instruction_editing_llm_rejected.csv}"

EXTRA_ARGS=()
if [[ "${PHYSTABMOL_ALLOW_MISSING_PARAPHRASE_TAGS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-missing-tags)
fi

python3 -m phystabmol.instruction_paraphrases filter \
  --dataset "$DATASET" \
  --paraphrases "$PARAPHRASES" \
  --out "$OUT" \
  --rejected-out "$REJECTED" \
  "${EXTRA_ARGS[@]}"

echo "Verified LLM paraphrases written to $OUT"
echo "Rejected LLM paraphrases written to $REJECTED"
