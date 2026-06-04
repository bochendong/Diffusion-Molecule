#!/usr/bin/env bash
# Compute Canada environment for RDKit-backed pair mining/evaluation.

set -euo pipefail

module load StdEnv/2023
module load python/3.11
module load rdkit/2025.09.4

python - <<'PY'
import sys
from rdkit import rdBase

print(f"Python: {sys.version.split()[0]}")
print(f"RDKit: {rdBase.rdkitVersion}")
PY
