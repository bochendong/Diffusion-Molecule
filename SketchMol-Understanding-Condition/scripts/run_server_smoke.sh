#!/usr/bin/env bash
# Run server-side smoke checks for SketchMol Understanding-Condition.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TORCH_PYTHON="${SUCC_TORCH_PYTHON:-/home/bdong/scratch/venvs/phystabmol/bin/python}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "SketchMol Understanding-Condition server smoke"
echo "  project_dir=$PROJECT_DIR"
echo "  torch_python=$TORCH_PYTHON"

echo "[1/4] Loading Compute Canada RDKit module"
module load StdEnv/2023
module load python/3.11
module load rdkit/2025.09.4

echo "[2/4] Verifying RDKit-backed chemistry helpers"
python - <<'PY'
from rdkit import rdBase
from sketchmol_understanding_condition.chem import (
    canonical_smiles,
    morgan_tanimoto,
    scaffold_smiles,
)

print(f"RDKit version: {rdBase.rdkitVersion}")
source = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
target = "COc1ccc2nc(S(N)(=O)=O)sc2c1"
assert canonical_smiles(source) is not None
assert scaffold_smiles(source) == scaffold_smiles(target)
similarity = morgan_tanimoto(source, target)
assert similarity is not None and 0.0 <= similarity <= 1.0
print(f"chemistry smoke ok: tanimoto={similarity:.4f}")
PY

echo "[3/4] Compiling Python files"
python -m compileall sketchmol_understanding_condition scripts tests

echo "[4/4] Verifying Torch condition encoder"
"$TORCH_PYTHON" - <<'PY'
import pathlib
import sys

project_dir = pathlib.Path.cwd()
sys.path.insert(0, str(project_dir))

import torch
from sketchmol_understanding_condition.encoders import (
    HybridConditionEncoder,
    MolecularQueryProjector,
)

print(f"Torch version: {torch.__version__}")
projector = MolecularQueryProjector(
    mllm_hidden_dim=16,
    context_dim=8,
    num_queries=4,
    hidden_dim=32,
)
hidden = torch.randn(2, 5, 16)
mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
out = projector(hidden, mask)
assert out.tokens.shape == (2, 4, 8)
assert out.attention_mask.shape == (2, 4)

encoder = HybridConditionEncoder(projector)
property_tokens = torch.randn(2, 3, 8)
hybrid = encoder(hidden, property_tokens=property_tokens)
assert hybrid.tokens.shape == (2, 7, 8)
assert hybrid.attention_mask.shape == (2, 7)
print("encoder smoke ok")
PY

echo "SketchMol Understanding-Condition server smoke finished."
