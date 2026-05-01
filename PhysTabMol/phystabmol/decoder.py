"""Physics-aware table-to-SMILES decoder.

This decoder is intentionally conservative. It assembles candidates from
drug-like scaffolds and functional group templates, then filters/ranks them by
descriptor agreement and simple medicinal chemistry rules. If RDKit is present,
descriptor/validity checks automatically become RDKit-backed via chem.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib

import numpy as np

from .chem import molecular_descriptors, passes_druglike_filters
from .schema import INTEGER_COLUMNS, TARGET_COLUMNS, TABLE_COLUMNS


DRUGLIKE_SOFT_PENALTY = 0.75


SCAFFOLDS = {
    0: ["CCC", "CCCC", "CC(C)C", "CC(C)CC", "CCCCCC", "CCOC(=O)C"],
    1: ["c1ccccc1", "Cc1ccccc1", "COc1ccccc1", "Nc1ccccc1", "Oc1ccccc1"],
    2: ["c1ccncc1", "COc1ccncc1", "Nc1ccncc1", "Cc1ccncc1"],
    3: ["c1ccoc1", "c1ccsc1", "COc1ccoc1", "Cc1ccsc1"],
    4: ["c1ccc2ccccc2c1", "c1ccc2ncccc2c1", "COc1ccc2ccccc2c1"],
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


def decode_table_row(row: dict[str, float], top_k: int = 5, seed: int = 0) -> list[DecodedCandidate]:
    scaffold_class = int(row.get("scaffold_class", 1))
    scaffolds = SCAFFOLDS.get(scaffold_class, SCAFFOLDS[1])
    candidates = set(scaffolds)
    for scaffold in scaffolds:
        candidates.update(_attach_groups(scaffold, row))
        candidates.update(_chain_variants(scaffold, row))

    decoded_by_smiles: dict[str, DecodedCandidate] = {}
    for candidate in _select_library_candidates(row, scaffold_class, seed=seed):
        score = candidate.score
        decoded_by_smiles[candidate.smiles] = DecodedCandidate(
            smiles=candidate.smiles,
            score=float(score),
            valid=candidate.valid,
            descriptors=candidate.descriptors,
            source=candidate.source,
        )

    for smi in candidates:
        rec = molecular_descriptors(smi)
        if not rec.valid:
            continue
        score = _rank_score(row, rec.descriptors, rec.smiles, seed)
        existing = decoded_by_smiles.get(rec.smiles)
        if existing is not None and existing.score <= score:
            continue
        decoded_by_smiles[rec.smiles] = (
            DecodedCandidate(
                smiles=rec.smiles,
                score=float(score),
                valid=rec.valid,
                descriptors=rec.descriptors,
                source="physics_aware_dynamic_decoder",
            )
        )
    decoded = list(decoded_by_smiles.values())
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


def _rank_score(target: dict[str, float], actual: dict[str, float], smiles: str, seed: int) -> float:
    score = _descriptor_distance(target, actual) + 0.22 * _count_distance(target, actual)
    if not passes_druglike_filters(actual):
        score += DRUGLIKE_SOFT_PENALTY
    return float(score + 0.025 * _stable_noise(smiles, seed))


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


def _count_distance(target: dict[str, float], actual: dict[str, float]) -> float:
    diffs = []
    for col in TABLE_COLUMNS[8:]:
        if col not in INTEGER_COLUMNS:
            continue
        scale = 8.0 if col in {"C"} else 3.0
        diffs.append(abs(float(target.get(col, 0.0)) - float(actual.get(col, 0.0))) / scale)
    return float(np.mean(diffs)) if diffs else 0.0


def _stable_noise(smiles: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _select_library_candidates(
    row: dict[str, float],
    scaffold_class: int,
    seed: int,
    limit: int = 96,
) -> list[DecodedCandidate]:
    candidates, targets, counts, drug_penalties = _candidate_index_for(scaffold_class)
    if not candidates:
        return []
    target_vec = np.asarray([float(row.get(col, 0.0)) for col in TARGET_COLUMNS], dtype=np.float32)
    count_cols = TABLE_COLUMNS[8:]
    count_vec = np.asarray([float(row.get(col, 0.0)) for col in count_cols], dtype=np.float32)
    target_scales = np.asarray([120.0, 3.0, 0.35, 60.0, 3.0, 5.0, 5.0, 2.5], dtype=np.float32)
    count_scales = np.asarray([8.0 if col == "C" else 3.0 for col in count_cols], dtype=np.float32)
    scores = np.mean(np.abs(targets - target_vec) / target_scales, axis=1)
    scores += 0.22 * np.mean(np.abs(counts - count_vec) / count_scales, axis=1)
    scores += drug_penalties
    k = min(limit, len(candidates))
    chosen = np.argpartition(scores, k - 1)[:k]
    out = []
    for idx in chosen:
        candidate = candidates[int(idx)]
        out.append(
            DecodedCandidate(
                smiles=candidate.smiles,
                score=float(scores[idx] + 0.025 * _stable_noise(candidate.smiles, seed)),
                valid=candidate.valid,
                descriptors=candidate.descriptors,
                source=candidate.source,
            )
        )
    return out


@lru_cache(maxsize=8)
def _candidate_index_for(scaffold_class: int):
    library = _candidate_library()
    preferred = {scaffold_class}
    if scaffold_class in {1, 2, 3}:
        preferred.update({1, 2, 3})
    elif scaffold_class == 4:
        preferred.update({1, 4})
    else:
        preferred.update({0, 1})
    candidates = [candidate for candidate in library if int(candidate.descriptors.get("scaffold_class", 1)) in preferred]
    targets = np.asarray(
        [[float(candidate.descriptors.get(col, 0.0)) for col in TARGET_COLUMNS] for candidate in candidates],
        dtype=np.float32,
    )
    count_cols = TABLE_COLUMNS[8:]
    counts = np.asarray(
        [[float(candidate.descriptors.get(col, 0.0)) for col in count_cols] for candidate in candidates],
        dtype=np.float32,
    )
    drug_penalties = np.asarray(
        [0.0 if passes_druglike_filters(candidate.descriptors) else DRUGLIKE_SOFT_PENALTY for candidate in candidates],
        dtype=np.float32,
    )
    return candidates, targets, counts, drug_penalties


@lru_cache(maxsize=1)
def _candidate_library() -> tuple[DecodedCandidate, ...]:
    decoded = []
    for smi in sorted(_library_smiles()):
        rec = molecular_descriptors(smi)
        if not rec.valid:
            continue
        decoded.append(
            DecodedCandidate(
                smiles=rec.smiles,
                score=0.0,
                valid=True,
                descriptors=rec.descriptors,
                source="physics_aware_library_decoder",
            )
        )
    by_smiles = {candidate.smiles: candidate for candidate in decoded}
    return tuple(by_smiles.values())


def _library_smiles() -> set[str]:
    smiles = set()
    for n in range(1, 25):
        chain = "C" * n
        smiles.update(
            {
                chain,
                chain + "O",
                chain + "N",
                chain + "C(=O)O",
                chain + "C(=O)N",
                chain + "OC(=O)C",
                chain + "S(=O)(=O)N",
            }
        )

    cores = [
        "c1ccccc1",
        "c1ccncc1",
        "c1ccoc1",
        "c1ccsc1",
        "c1ccc2ccccc2c1",
        "c1ccc2ncccc2c1",
    ]
    prefixes = ["", "C", "CC", "CCC", "CO", "CCO", "CN", "CCN", "O", "N", "F", "Cl", "Br", "CC(=O)N", "NC(=O)", "COC(=O)"]
    suffixes = ["", "C", "CC", "O", "N", "F", "Cl", "Br", "C(=O)O", "C(=O)N", "OC", "CN"]
    combo_prefixes = ["CC", "CO", "N", "F", "Cl", "CC(=O)N", "NC(=O)"]
    combo_suffixes = ["C", "O", "F", "Cl", "C(=O)O", "C(=O)N"]
    for core in cores:
        smiles.add(core)
        for prefix in prefixes:
            smiles.add(prefix + core)
        for suffix in suffixes:
            smiles.add(core + suffix)
        for prefix in combo_prefixes:
            for suffix in combo_suffixes:
                smiles.add(prefix + core + suffix)
        for n in range(4, 17, 3):
            chain = "C" * n
            smiles.add(chain + core)
            smiles.add(chain + core + "C(=O)N")
            smiles.add(chain + "O" + core)
            smiles.add(chain + "NC(=O)" + core)

    smiles.update(
        {
            "O1CCNCC1",
            "N1CCCCC1",
            "CN1CCCCC1",
            "O=C(NC1CCCCC1)c1ccccc1",
            "CCN(CC)CC",
            "CCOC(=O)c1ccccc1",
            "CC(=O)Nc1ccc(O)cc1",
            "COc1ccc(C(=O)N)cc1",
            "Nc1ncccn1",
            "COc1ncccc1",
            "CCn1ccnc1",
            "CCOc1ccccc1F",
            "Clc1ccc(C(=O)N)cc1",
            "Fc1ccc(C(=O)O)cc1",
        }
    )
    return smiles
