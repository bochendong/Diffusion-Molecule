#!/usr/bin/env bash
# Load cluster RDKit modules and verify PyTDC oracles before MolEdit table metrics.

set -euo pipefail

_init_modules() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
}

ensure_moledit_oracle_env() {
  local python_bin="${1:?python bin required}"
  if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: python bin is not executable: $python_bin" >&2
    return 1
  fi
  local modules="${SUCC_UNIFIED_MOLEDIT_RDKIT_MODULES:-gcc/12.3 rdkit/2024.09.6}"

  _init_modules
  if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    # shellcheck disable=SC2086
    module load $modules
    echo "Loaded MolEdit oracle modules: $modules"
  else
    echo "WARNING: module command unavailable; relying on venv RDKit for MolEdit oracles" >&2
  fi

  "$python_bin" - <<'PY'
import sys
import types


def _ensure_rdkit_six_compat() -> None:
    if "rdkit.six" in sys.modules:
        return
    try:
        from rdkit.six import iteritems  # noqa: F401
    except ModuleNotFoundError:
        six_mod = types.ModuleType("rdkit.six")
        six_mod.iteritems = dict.items
        sys.modules["rdkit.six"] = six_mod


_ensure_rdkit_six_compat()
from tdc import Oracle

score = float(Oracle(name="DRD2")("CCO"))
print(f"moledit_oracle_preflight_ok DRD2(CCO)={score:.6f}")
PY
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ensure_moledit_oracle_env "${SUCC_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
fi
