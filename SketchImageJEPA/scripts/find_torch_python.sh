#!/usr/bin/env bash
# List candidate Python interpreters and whether they can import torch/RDKit.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${SKETCHIMAGE_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $SKETCHIMAGE_MODULES
fi

CANDIDATES=()
if [[ -n "${SKETCHIMAGE_PYTHON_BIN:-}" ]]; then
  CANDIDATES+=("$SKETCHIMAGE_PYTHON_BIN")
fi
for path in \
  /scratch/bdong/venvs/*/bin/python \
  /home/bdong/scratch/venvs/*/bin/python \
  "$(command -v python3 2>/dev/null || true)"
do
  if [[ -n "$path" && -x "$path" ]]; then
    CANDIDATES+=("$path")
  fi
done

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo "No Python candidates found." >&2
  exit 2
fi

printf "%-64s %-8s %-8s %s\n" "python" "torch" "rdkit" "details"
seen=""
for python_bin in "${CANDIDATES[@]}"; do
  if [[ "$seen" == *"|$python_bin|"* ]]; then
    continue
  fi
  seen="${seen}|${python_bin}|"
  "$python_bin" - <<'PY' "$python_bin"
import sys

python_bin = sys.argv[1]

def probe(module_name):
    try:
        module = __import__(module_name)
        return "yes", getattr(module, "__version__", "ok")
    except Exception as exc:
        return "no", str(exc).splitlines()[0]

torch_ok, torch_detail = probe("torch")
rdkit_ok, rdkit_detail = probe("rdkit")
detail = f"torch={torch_detail}; rdkit={rdkit_detail}"
print(f"{python_bin:<64} {torch_ok:<8} {rdkit_ok:<8} {detail}")
PY
done
