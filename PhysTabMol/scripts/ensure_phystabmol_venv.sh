#!/bin/bash
# Auto-activate PhysTabMol venv when present (default: scratch/venvs/phystabmol).
# Override: export PHYSTABMOL_VENV=/path/to/venv
# Expects PHYSTABMOL_ROOT to be set and is meant to be sourced after `cd "$PHYSTABMOL_ROOT"`.

if [[ -n "${PHYSTABMOL_VENV_ACTIVE:-}" ]]; then
  return 0
fi

PHYSTABMOL_VENV_DIR="${PHYSTABMOL_VENV:-}"
if [[ -z "$PHYSTABMOL_VENV_DIR" ]]; then
  _parent="$(cd "$PHYSTABMOL_ROOT/../../.." && pwd)"
  if [[ -f "$_parent/venvs/phystabmol/bin/activate" ]]; then
    PHYSTABMOL_VENV_DIR="$_parent/venvs/phystabmol"
  fi
  unset _parent
fi

if [[ -n "$PHYSTABMOL_VENV_DIR" && -f "$PHYSTABMOL_VENV_DIR/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$PHYSTABMOL_VENV_DIR/bin/activate"
  export PHYSTABMOL_VENV_ACTIVE=1
fi
