"""Dataset construction for edit, inpainting, and de novo smoke experiments."""

from __future__ import annotations

import csv
from pathlib import Path

from .chem import molecular_descriptors
from .schema import GenerationRequest, TaskType


SMOKE_SMILES = [
    "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
    "CC(=O)Nc1ccc(O)cc1",
    "COc1ccc(CCN)cc1",
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "CC(C)NCC(O)COc1ccccc1",
    "O=C(Nc1ccccc1)c1ccccc1",
]


def load_smiles_csv(path: str | Path | None, limit: int = 0) -> list[str]:
    if path is None or not str(path).strip() or not Path(path).exists() or Path(path).is_dir():
        return list(SMOKE_SMILES)
    out = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles = row.get("smiles") or row.get("SMILES") or row.get("canonical_smiles")
            if not smiles:
                continue
            rec = molecular_descriptors(smiles)
            if rec.valid:
                out.append(rec.smiles)
            if limit and len(out) >= limit:
                break
    return out or list(SMOKE_SMILES)


def build_smoke_requests(smiles: list[str]) -> list[tuple[GenerationRequest, str]]:
    requests: list[tuple[GenerationRequest, str]] = []
    for idx, smi in enumerate(smiles):
        target = smiles[(idx + 1) % len(smiles)]
        requests.append(
            (
                GenerationRequest(
                    task_type=TaskType.EDIT,
                    source_smiles=smi,
                    instruction="The molecule is too lipophilic. Lower LogP and keep the core similar.",
                ),
                target,
            )
        )
        requests.append(
            (
                GenerationRequest(
                    task_type=TaskType.INPAINT,
                    source_smiles=smi,
                    mask_smarts="[*:1]",
                    instruction="Fill the masked substituent to improve solubility while preserving the scaffold.",
                ),
                target,
            )
        )
        requests.append(
            (
                GenerationRequest(
                    task_type=TaskType.DE_NOVO,
                    instruction="Generate a CNS-like drug-like molecule with MW below 450 and TPSA below 90.",
                ),
                smi,
            )
        )
    return requests
