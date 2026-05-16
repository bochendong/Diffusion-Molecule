#!/bin/bash
set -euo pipefail

PHYSTABMOL_ROOT="${PHYSTABMOL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PHYSTABMOL_ROOT"
if command -v module >/dev/null 2>&1 && [[ "${PHYSTABMOL_USE_MODULE_ENV:-1}" == "1" && -f "$PHYSTABMOL_ROOT/scripts/env_module_venv.sh" ]]; then
  # shellcheck source=/dev/null
  source "$PHYSTABMOL_ROOT/scripts/env_module_venv.sh"
else
  # shellcheck source=/dev/null
  source "$PHYSTABMOL_ROOT/scripts/ensure_phystabmol_venv.sh"
fi

python3 -m phystabmol.instruction_report \
  --runs-root "${PHYSTABMOL_RUNS_ROOT:-runs}" \
  --out "${PHYSTABMOL_INSTRUCTION_SUMMARY_OUT:-outputs/instruction_paper_summary.csv}" \
  --latex-out "${PHYSTABMOL_INSTRUCTION_LATEX_OUT:-outputs/instruction_paper_summary.tex}" \
  --breakdown-out "${PHYSTABMOL_INSTRUCTION_BREAKDOWN_OUT:-outputs/instruction_failure_breakdown.csv}"
