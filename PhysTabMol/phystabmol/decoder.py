"""Physics-aware table-to-SMILES decoder.

This decoder is intentionally conservative. It assembles candidates from
drug-like scaffolds and functional group templates, then filters/ranks them by
descriptor agreement and simple medicinal chemistry rules. If RDKit is present,
descriptor/validity checks automatically become RDKit-backed via chem.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chem import molecular_descriptors, passes_druglike_filters
from .schema import TARGET_COLUMNS


SCAFFOLDS = {
    0: ["CCC", "CCCC", "CC(C)C", "CC(C)CC"],
    1: ["c1ccccc1", "Cc1ccccc1", "COc1ccccc1"],
    2: ["c1ccncc1", "COc1ccncc1", "Nc1ccncc1"],
    3: ["c1ccoc1", "c1ccsc1", "COc1ccoc1"],
    4: ["c1ccc2ccccc2c1", "c1ccc2ncccc2c1"],
}

GROUP_PREFIXES = {
    "fg_ester": ["CC(=O)O", "CCOC(=O)"],
    "fg_amide": ["CC(=O)N", "NC(=O)"],
    "fg_amine": ["N", "CN", "CCN"],
    "fg_alcohol": ["O", "CO", "CCO"],
    "fg_halogen": ["F", "Cl", "Br"],
}


@dataclass
class DecodedCandidate:
    smiles: str
    score: float
    valid: bool
    descriptors: dict[str, float]
    source: str


def decode_table_row(row: dict[str, float], top_k: int = 5) -> list[DecodedCandidate]:
    scaffold_class = int(row.get("scaffold_class", 1))
    scaffolds = SCAFFOLDS.get(scaffold_class, SCAFFOLDS[1])
    candidates = set(scaffolds)
    for scaffold in scaffolds:
        candidates.update(_attach_groups(scaffold, row))
        candidates.update(_chain_variants(scaffold, row))

    decoded = []
    for smi in candidates:
        rec = molecular_descriptors(smi)
        if not rec.valid:
            continue
        score = _descriptor_distance(row, rec.descriptors)
        if not passes_druglike_filters(rec.descriptors):
            score += 2.0
        decoded.append(
            DecodedCandidate(
                smiles=rec.smiles,
                score=float(score),
                valid=rec.valid,
                descriptors=rec.descriptors,
                source="physics_aware_template_decoder",
            )
        )
    decoded.sort(key=lambda c: c.score)
    return decoded[:top_k]


def _attach_groups(scaffold: str, row: dict[str, float]) -> set[str]:
    out = set()
    for group, prefixes in GROUP_PREFIXES.items():
        count = int(row.get(group, 0))
        if count <= 0:
            continue
        for prefix in prefixes[: min(2, count + 1)]:
            out.add(prefix + scaffold)
            if group in {"fg_amine", "fg_alcohol", "fg_halogen"}:
                out.add(scaffold + prefix)
    if row.get("N", 0) >= 1 and row.get("O", 0) >= 1:
        out.add("O=C(N)" + scaffold)
    if row.get("O", 0) >= 2:
        out.add("COC(=O)" + scaffold)
    return out


def _chain_variants(scaffold: str, row: dict[str, float]) -> set[str]:
    c_count = int(row.get("C", 6))
    chain_len = int(np.clip(c_count // 6, 1, 5))
    chain = "C" * chain_len
    variants = {chain + scaffold, scaffold + chain}
    if row.get("HBA", 0) >= 2:
        variants.add(chain + "O" + scaffold)
    if row.get("HBD", 0) >= 1:
        variants.add(chain + "N" + scaffold)
    return variants


def _descriptor_distance(target: dict[str, float], actual: dict[str, float]) -> float:
    scales = {
        "MW": 120.0,
        "LogP": 3.0,
        "QED": 0.35,
        "TPSA": 60.0,
        "HBD": 3.0,
        "HBA": 5.0,
        "RB": 5.0,
        "SA": 2.5,
    }
    diffs = []
    for col in TARGET_COLUMNS:
        diffs.append(abs(float(target.get(col, 0.0)) - float(actual.get(col, 0.0))) / scales[col])
    return float(np.mean(diffs))
