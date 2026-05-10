"""Retrieval and MMP-like decoder for tabular molecular plans.

The physics decoder is deliberately small and valid, but it collapses many
table rows to a tiny set of template molecules. This module expands the decode
space with real training molecules plus simple RDKit-verified local edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os

import numpy as np
import pandas as pd

if os.environ.get("PHYSTABMOL_DISABLE_SKLEARN_NN", "0") == "1":  # pragma: no cover - local smoke path.
    NearestNeighbors = None
    SKLEARN_NN_AVAILABLE = False
else:
    try:  # pragma: no cover - server path.
        from sklearn.neighbors import NearestNeighbors

        SKLEARN_NN_AVAILABLE = True
    except Exception:  # pragma: no cover
        NearestNeighbors = None
        SKLEARN_NN_AVAILABLE = False

from . import chem as chem_mod
from .chem import canonicalize_smiles, molecular_descriptors, passes_druglike_filters, tanimoto
from .decoder import DRUGLIKE_SOFT_PENALTY, DecodedCandidate, decode_table_row
from .mmp_transform_decoder import MMPTransformConfig, MMPTransformIndex, decode_mmp_table_row
from .schema import INTEGER_COLUMNS, TABLE_COLUMNS, TARGET_COLUMNS


TABLE_SCALE_BY_COLUMN = {
    "MW": 120.0,
    "LogP": 3.0,
    "QED": 0.35,
    "TPSA": 60.0,
    "HBD": 3.0,
    "HBA": 5.0,
    "RB": 5.0,
    "SA": 2.5,
    "C": 8.0,
    "N": 3.0,
    "O": 3.0,
    "S": 2.0,
    "F": 3.0,
    "Cl": 2.0,
    "Br": 2.0,
    "I": 1.0,
    "ring_count": 3.0,
    "scaffold_class": 4.0,
    "fg_ester": 3.0,
    "fg_amide": 3.0,
    "fg_amine": 3.0,
    "fg_alcohol": 4.0,
    "fg_halogen": 3.0,
}
TABLE_SCALES = np.asarray([TABLE_SCALE_BY_COLUMN[col] for col in TABLE_COLUMNS], dtype=np.float32)


@dataclass(frozen=True)
class RetrievalDecoderConfig:
    neighbors: int = 256
    edit_neighbors: int = 12
    max_candidates: int = 100000
    exact_train_penalty: float = 0.08
    edit_bonus: float = 0.08
    prompt_match_bonus: float = 1.5
    prompt_miss_penalty: float = 4.0


class RetrievalCandidateIndex:
    def __init__(self, candidates: list[DecodedCandidate], table_x: np.ndarray):
        if not candidates:
            raise ValueError("Retrieval decoder needs at least one candidate molecule.")
        self.candidates = candidates
        self.table_x = np.asarray(table_x, dtype=np.float32)
        self.scaled_x = self.table_x / TABLE_SCALES
        self.train_smiles = {candidate.smiles for candidate in candidates}
        self.nn = None
        if SKLEARN_NN_AVAILABLE:
            n_neighbors = min(len(self.candidates), 1024)
            self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(self.scaled_x)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, max_candidates: int = 100000) -> "RetrievalCandidateIndex":
        by_smiles: dict[str, DecodedCandidate] = {}
        rows = []
        source = df.dropna(subset=["smiles"]).head(max_candidates)
        for _, row in source.iterrows():
            can = canonicalize_smiles(str(row["smiles"]))
            if can is None or can in by_smiles:
                continue
            descriptors = {col: float(row[col]) for col in TABLE_COLUMNS if col in row and not pd.isna(row[col])}
            if len(descriptors) < len(TABLE_COLUMNS):
                rec = molecular_descriptors(can)
                if not rec.valid:
                    continue
                descriptors.update(rec.descriptors)
                can = rec.smiles
            if can in by_smiles:
                continue
            candidate = DecodedCandidate(
                smiles=can,
                score=0.0,
                valid=True,
                descriptors={col: float(descriptors.get(col, 0.0)) for col in TABLE_COLUMNS},
                source="retrieval_train_pool_decoder",
            )
            by_smiles[can] = candidate
            rows.append([float(candidate.descriptors.get(col, 0.0)) for col in TABLE_COLUMNS])
        return cls(list(by_smiles.values()), np.asarray(rows, dtype=np.float32))

    def decode(
        self,
        table_row: dict[str, float],
        top_k: int,
        seed: int,
        config: RetrievalDecoderConfig,
        prompt_smiles: str | None = None,
    ) -> list[DecodedCandidate]:
        indices = self._nearest(_table_vector(table_row), config.neighbors)
        decoded_by_smiles: dict[str, DecodedCandidate] = {}
        for idx in indices:
            base = self.candidates[int(idx)]
            score = _retrieval_score(table_row, base.descriptors, base.smiles, seed, config, prompt_smiles, exact_train=True, edited=False)
            decoded_by_smiles[base.smiles] = DecodedCandidate(
                smiles=base.smiles,
                score=score,
                valid=True,
                descriptors=base.descriptors,
                source="retrieval_train_pool_decoder",
            )

        for idx in indices[: max(0, config.edit_neighbors)]:
            base = self.candidates[int(idx)]
            for smi in _mmp_like_variants(base.smiles, table_row):
                rec = molecular_descriptors(smi)
                if not rec.valid:
                    continue
                score = _retrieval_score(table_row, rec.descriptors, rec.smiles, seed, config, prompt_smiles, exact_train=rec.smiles in self.train_smiles, edited=True)
                existing = decoded_by_smiles.get(rec.smiles)
                if existing is None or score < existing.score:
                    decoded_by_smiles[rec.smiles] = DecodedCandidate(
                        smiles=rec.smiles,
                        score=score,
                        valid=True,
                        descriptors=rec.descriptors,
                        source="retrieval_mmp_edit_decoder",
                    )

        decoded = list(decoded_by_smiles.values())
        decoded.sort(key=lambda item: item.score)
        return decoded[:top_k]

    def _nearest(self, raw_vec: np.ndarray, n: int) -> list[int]:
        k = min(max(1, int(n)), len(self.candidates))
        scaled = raw_vec / TABLE_SCALES
        if self.nn is not None:
            _, indices = self.nn.kneighbors(scaled[None, :], n_neighbors=k)
            return [int(idx) for idx in indices[0]]
        distances = np.mean(np.abs(self.scaled_x - scaled), axis=1)
        chosen = np.argpartition(distances, k - 1)[:k]
        return [int(idx) for idx in chosen[np.argsort(distances[chosen])]]


def decode_retrieval_table_row(
    row: dict[str, float],
    top_k: int,
    seed: int,
    mode: str,
    index: RetrievalCandidateIndex | None,
    config: RetrievalDecoderConfig | None = None,
    include_dynamic: bool = True,
    prompt_smiles: str | None = None,
    mmp_index: MMPTransformIndex | None = None,
    mmp_config: MMPTransformConfig | None = None,
) -> list[DecodedCandidate]:
    config = config or RetrievalDecoderConfig()
    candidates: list[DecodedCandidate] = []
    if mode in {"mmp", "hybrid_mmp"} and mmp_index is not None:
        candidates.extend(
            decode_mmp_table_row(
                row,
                top_k=max(top_k * 8, (mmp_config or MMPTransformConfig()).target_neighbors // 4),
                seed=seed,
                index=mmp_index,
                config=mmp_config,
                prompt_smiles=prompt_smiles,
            )
        )
    if mode in {"retrieval", "hybrid", "hybrid_mmp"} and index is not None:
        candidates.extend(index.decode(row, top_k=max(top_k * 8, config.neighbors // 4), seed=seed, config=config, prompt_smiles=prompt_smiles))
    if mode in {"physics", "hybrid", "hybrid_mmp"}:
        candidates.extend(decode_table_row(row, top_k=max(top_k * 8, top_k), seed=seed, include_dynamic=include_dynamic))
    if not candidates:
        candidates.extend(decode_table_row(row, top_k=top_k, seed=seed, include_dynamic=include_dynamic))
    return _dedupe_rank(candidates, top_k=top_k)


def _retrieval_score(
    target: dict[str, float],
    actual: dict[str, float],
    smiles: str,
    seed: int,
    config: RetrievalDecoderConfig,
    prompt_smiles: str | None,
    exact_train: bool,
    edited: bool,
) -> float:
    score = _descriptor_distance(target, actual) + 0.20 * _count_distance(target, actual)
    if not passes_druglike_filters(actual):
        score += DRUGLIKE_SOFT_PENALTY
    if exact_train:
        score += config.exact_train_penalty
    if edited:
        score -= config.edit_bonus
    if prompt_smiles:
        if _contains_prompt(prompt_smiles, smiles):
            score -= config.prompt_match_bonus
        else:
            score += config.prompt_miss_penalty
    score += 0.015 * _stable_noise(smiles, seed)
    return float(score)


def _descriptor_distance(target: dict[str, float], actual: dict[str, float]) -> float:
    scales = {"MW": 120.0, "LogP": 3.0, "QED": 0.35, "TPSA": 60.0, "HBD": 3.0, "HBA": 5.0, "RB": 5.0, "SA": 2.5}
    return float(np.mean([abs(float(target.get(col, 0.0)) - float(actual.get(col, 0.0))) / scales[col] for col in TARGET_COLUMNS]))


def _count_distance(target: dict[str, float], actual: dict[str, float]) -> float:
    diffs = []
    for col in TABLE_COLUMNS[8:]:
        if col not in INTEGER_COLUMNS:
            continue
        scale = 8.0 if col == "C" else 3.0
        diffs.append(abs(float(target.get(col, 0.0)) - float(actual.get(col, 0.0))) / scale)
    return float(np.mean(diffs)) if diffs else 0.0


def _table_vector(table_row: dict[str, float]) -> np.ndarray:
    return np.asarray([float(table_row.get(col, 0.0)) for col in TABLE_COLUMNS], dtype=np.float32)


def _dedupe_rank(candidates: list[DecodedCandidate], top_k: int) -> list[DecodedCandidate]:
    by_smiles: dict[str, DecodedCandidate] = {}
    for candidate in candidates:
        existing = by_smiles.get(candidate.smiles)
        if existing is None or candidate.score < existing.score:
            by_smiles[candidate.smiles] = candidate
    decoded = list(by_smiles.values())
    decoded.sort(key=lambda item: item.score)
    return decoded[:top_k]


def _mmp_like_variants(smiles: str, target: dict[str, float]) -> set[str]:
    variants = set()
    rec = molecular_descriptors(smiles)
    if not rec.valid:
        return variants
    actual = rec.descriptors
    additions = []
    if float(target.get("MW", 0.0)) > float(actual.get("MW", 0.0)) + 20 or float(target.get("C", 0.0)) > float(actual.get("C", 0.0)) + 1:
        additions.extend(["methyl", "ethyl"])
    if float(target.get("LogP", 0.0)) > float(actual.get("LogP", 0.0)) + 0.4 or float(target.get("fg_halogen", 0.0)) > float(actual.get("fg_halogen", 0.0)):
        additions.extend(["fluoro", "chloro", "methyl"])
    if float(target.get("TPSA", 0.0)) > float(actual.get("TPSA", 0.0)) + 15 or float(target.get("HBA", 0.0)) > float(actual.get("HBA", 0.0)):
        additions.extend(["hydroxy", "amide", "carboxyl"])
    if float(target.get("HBD", 0.0)) > float(actual.get("HBD", 0.0)):
        additions.extend(["amino", "hydroxy"])
    if float(target.get("TPSA", 0.0)) < float(actual.get("TPSA", 0.0)) - 20:
        variants.update(_remove_terminal_heteroatoms(smiles))
    for group in list(dict.fromkeys(additions))[:5]:
        variants.update(_add_group_variants(smiles, group, limit=6))
    return variants


def _add_group_variants(smiles: str, group: str, limit: int) -> set[str]:
    if not chem_mod.RDKIT_AVAILABLE:
        return _fallback_group_variants(smiles, group)
    mol = chem_mod.Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    out = set()
    for atom in _attachment_atoms(mol)[:limit]:
        variant = _try_add_group(mol, atom, group)
        if variant:
            out.add(variant)
    return out


def _attachment_atoms(mol) -> list[int]:
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in {1, 9, 17, 35, 53}:
            continue
        if atom.GetTotalNumHs() > 0:
            atoms.append(int(atom.GetIdx()))
    return atoms


def _try_add_group(mol, atom_idx: int, group: str) -> str | None:
    rw = chem_mod.Chem.RWMol(mol)
    try:
        if group == "methyl":
            c = rw.AddAtom(chem_mod.Chem.Atom("C"))
            rw.AddBond(atom_idx, c, chem_mod.Chem.BondType.SINGLE)
        elif group == "ethyl":
            c1 = rw.AddAtom(chem_mod.Chem.Atom("C"))
            c2 = rw.AddAtom(chem_mod.Chem.Atom("C"))
            rw.AddBond(atom_idx, c1, chem_mod.Chem.BondType.SINGLE)
            rw.AddBond(c1, c2, chem_mod.Chem.BondType.SINGLE)
        elif group == "fluoro":
            f = rw.AddAtom(chem_mod.Chem.Atom("F"))
            rw.AddBond(atom_idx, f, chem_mod.Chem.BondType.SINGLE)
        elif group == "chloro":
            cl = rw.AddAtom(chem_mod.Chem.Atom("Cl"))
            rw.AddBond(atom_idx, cl, chem_mod.Chem.BondType.SINGLE)
        elif group == "hydroxy":
            o = rw.AddAtom(chem_mod.Chem.Atom("O"))
            rw.AddBond(atom_idx, o, chem_mod.Chem.BondType.SINGLE)
        elif group == "amino":
            n = rw.AddAtom(chem_mod.Chem.Atom("N"))
            rw.AddBond(atom_idx, n, chem_mod.Chem.BondType.SINGLE)
        elif group == "amide":
            c = rw.AddAtom(chem_mod.Chem.Atom("C"))
            o = rw.AddAtom(chem_mod.Chem.Atom("O"))
            n = rw.AddAtom(chem_mod.Chem.Atom("N"))
            rw.AddBond(atom_idx, c, chem_mod.Chem.BondType.SINGLE)
            rw.AddBond(c, o, chem_mod.Chem.BondType.DOUBLE)
            rw.AddBond(c, n, chem_mod.Chem.BondType.SINGLE)
        elif group == "carboxyl":
            c = rw.AddAtom(chem_mod.Chem.Atom("C"))
            o1 = rw.AddAtom(chem_mod.Chem.Atom("O"))
            o2 = rw.AddAtom(chem_mod.Chem.Atom("O"))
            rw.AddBond(atom_idx, c, chem_mod.Chem.BondType.SINGLE)
            rw.AddBond(c, o1, chem_mod.Chem.BondType.DOUBLE)
            rw.AddBond(c, o2, chem_mod.Chem.BondType.SINGLE)
        else:
            return None
        out = rw.GetMol()
        chem_mod.Chem.SanitizeMol(out)
        return chem_mod.Chem.MolToSmiles(out, canonical=True)
    except Exception:
        return None


def _remove_terminal_heteroatoms(smiles: str) -> set[str]:
    if not chem_mod.RDKIT_AVAILABLE:
        return set()
    mol = chem_mod.Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    out = set()
    for atom in mol.GetAtoms():
        if atom.GetDegree() != 1 or atom.GetAtomicNum() not in {7, 8, 9, 17, 35, 53}:
            continue
        rw = chem_mod.Chem.RWMol(mol)
        try:
            rw.RemoveAtom(int(atom.GetIdx()))
            candidate = rw.GetMol()
            chem_mod.Chem.SanitizeMol(candidate)
            out.add(chem_mod.Chem.MolToSmiles(candidate, canonical=True))
        except Exception:
            continue
    return out


def _fallback_group_variants(smiles: str, group: str) -> set[str]:
    suffix = {
        "methyl": "C",
        "ethyl": "CC",
        "fluoro": "F",
        "chloro": "Cl",
        "hydroxy": "O",
        "amino": "N",
        "amide": "C(=O)N",
        "carboxyl": "C(=O)O",
    }.get(group)
    return {smiles + suffix} if suffix else set()


@lru_cache(maxsize=500000)
def _contains_prompt(prompt_smiles: str, candidate_smiles: str) -> bool:
    prompt = canonicalize_smiles(prompt_smiles)
    candidate = canonicalize_smiles(candidate_smiles)
    if prompt is None or candidate is None:
        return False
    if prompt == candidate:
        return True
    if chem_mod.RDKIT_AVAILABLE:
        try:
            prompt_mol = chem_mod.Chem.MolFromSmiles(prompt)
            candidate_mol = chem_mod.Chem.MolFromSmiles(candidate)
            return bool(candidate_mol is not None and prompt_mol is not None and candidate_mol.HasSubstructMatch(prompt_mol))
        except Exception:
            return False
    return prompt in candidate or tanimoto(prompt, candidate) >= 0.5


def _stable_noise(smiles: str, seed: int) -> float:
    digest = hashlib.sha256(f"retrieval-decoder:{seed}:{smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32
