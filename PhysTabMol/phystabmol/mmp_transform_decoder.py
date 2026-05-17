"""Data-driven MMP/CReM-style decoder for tabular molecular plans.

The retrieval decoder can already find real molecules near a generated table
row and make a few hand-written edits. This module adds a learned transform
layer: it mines descriptor-neighbor molecular pairs and small attachable
fragments from the training split, then uses them as a constrained decoder.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if os.environ.get("PHYSTABMOL_DISABLE_SKLEARN_NN", "0") == "1":  # pragma: no cover
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
from .decoder import DRUGLIKE_SOFT_PENALTY, DecodedCandidate
from .progress import iter_progress
from .schema import TABLE_COLUMNS, TARGET_COLUMNS


TARGET_SCALES = np.asarray([120.0, 3.0, 0.35, 60.0, 3.0, 5.0, 5.0, 2.5], dtype=np.float32)
COUNT_SCALES = np.asarray([8.0, 3.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.0, 3.0, 4.0, 3.0, 3.0, 3.0, 4.0, 3.0], dtype=np.float32)
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
class MMPTransformConfig:
    max_pairs: int = 80000
    pairs_per_source: int = 6
    min_pair_similarity: float = 0.25
    max_pair_similarity: float = 0.98
    target_neighbors: int = 384
    delta_neighbors: int = 256
    source_neighbors: int = 128
    fragment_neighbors: int = 12
    attachment_limit: int = 4
    max_fragments: int = 12000
    max_fragment_atoms: int = 8
    exact_train_penalty: float = 0.30
    transform_bonus: float = 0.10
    fragment_bonus: float = 0.16
    prompt_match_bonus: float = 2.5
    prompt_miss_penalty: float = 8.0
    fragment_exact_penalty: float = 0.18
    fragment_growth_steps: int = 2
    fragment_growth_beam_size: int = 12
    fragment_second_step_neighbors: int = 6
    fragment_growth_mw_gap: float = 25.0


@dataclass(frozen=True)
class MMPPair:
    source_smiles: str
    target_smiles: str
    source_vec: np.ndarray
    target_vec: np.ndarray
    delta_vec: np.ndarray
    target_descriptors: dict[str, float]
    edit_tags: str
    similarity: float


@dataclass(frozen=True)
class MMPFragment:
    fragment_smiles: str
    approx_delta: np.ndarray
    heavy_atoms: int
    frequency: int


class MMPTransformIndex:
    def __init__(self, pairs: list[MMPPair], fragments: list[MMPFragment], train_smiles: set[str]):
        if not pairs and not fragments:
            raise ValueError("MMP transform decoder needs at least one pair or fragment.")
        self.pairs = pairs
        self.fragments = fragments
        self.train_smiles = train_smiles
        self.target_x = np.asarray([item.target_vec for item in pairs], dtype=np.float32) if pairs else np.zeros((0, len(TABLE_COLUMNS)), dtype=np.float32)
        self.delta_x = np.asarray([item.delta_vec for item in pairs], dtype=np.float32) if pairs else np.zeros((0, len(TABLE_COLUMNS)), dtype=np.float32)
        self.source_x = np.asarray([item.source_vec for item in pairs], dtype=np.float32) if pairs else np.zeros((0, len(TABLE_COLUMNS)), dtype=np.float32)
        self.fragment_delta_x = (
            np.asarray([item.approx_delta for item in fragments], dtype=np.float32) if fragments else np.zeros((0, len(TABLE_COLUMNS)), dtype=np.float32)
        )
        self.target_nn = _fit_nn(self.target_x / TABLE_SCALES) if len(self.target_x) else None
        self.delta_nn = _fit_nn(self.delta_x / TABLE_SCALES) if len(self.delta_x) else None
        self.source_nn = _fit_nn(self.source_x / TABLE_SCALES) if len(self.source_x) else None
        self.fragment_nn = _fit_nn(self.fragment_delta_x / TABLE_SCALES) if len(self.fragment_delta_x) else None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, config: MMPTransformConfig | None = None) -> "MMPTransformIndex":
        config = config or MMPTransformConfig()
        records = _records_from_dataframe(df)
        pairs = mine_mmp_pairs(records, config=config)
        fragments = mine_fragments(records, config=config)
        return cls(pairs=pairs, fragments=fragments, train_smiles={item["smiles"] for item in records})

    @classmethod
    def from_library_csv(cls, path: str | Path) -> "MMPTransformIndex":
        df = pd.read_csv(path)
        pairs: list[MMPPair] = []
        fragments: list[MMPFragment] = []
        train_smiles: set[str] = set()
        pair_rows = df[df["record_type"] == "pair"] if "record_type" in df else df
        for _, row in iter_progress(pair_rows.iterrows(), total=len(pair_rows), label="loading MMP pair rows"):
            source = str(row.get("source_smiles", "")).strip()
            target = str(row.get("target_smiles", "")).strip()
            if not source or not target or source == "nan" or target == "nan":
                continue
            source_rec = molecular_descriptors(source)
            target_rec = molecular_descriptors(target)
            if not source_rec.valid or not target_rec.valid:
                continue
            source_vec = _descriptor_vector(source_rec.descriptors)
            target_vec = _descriptor_vector(target_rec.descriptors)
            pairs.append(
                MMPPair(
                    source_smiles=source_rec.smiles,
                    target_smiles=target_rec.smiles,
                    source_vec=source_vec,
                    target_vec=target_vec,
                    delta_vec=target_vec - source_vec,
                    target_descriptors=target_rec.descriptors,
                    edit_tags=str(row.get("edit_tags", "")),
                    similarity=_safe_float(row.get("similarity", 0.0)),
                )
            )
            train_smiles.update([source_rec.smiles, target_rec.smiles])
        if "record_type" in df:
            fragment_rows = df[df["record_type"] == "fragment"]
            for _, row in iter_progress(fragment_rows.iterrows(), total=len(fragment_rows), label="loading MMP fragment rows"):
                fragment = str(row.get("fragment_smiles", "")).strip()
                if not fragment or fragment == "nan":
                    continue
                fragments.append(
                    MMPFragment(
                        fragment_smiles=fragment,
                        approx_delta=np.asarray([_safe_float(row.get(f"frag_delta_{col}", 0.0)) for col in TABLE_COLUMNS], dtype=np.float32),
                        heavy_atoms=int(_safe_float(row.get("heavy_atoms", 0))),
                        frequency=max(1, int(_safe_float(row.get("frequency", 1), default=1.0))),
                    )
                )
        return cls(pairs=pairs, fragments=fragments, train_smiles=train_smiles)

    def decode(
        self,
        table_row: dict[str, float],
        top_k: int,
        seed: int,
        config: MMPTransformConfig,
        prompt_smiles: str | None = None,
    ) -> list[DecodedCandidate]:
        target_vec = _table_vector(table_row)
        prompt_vec = _smiles_vector(prompt_smiles)
        delta_vec = target_vec - prompt_vec if prompt_vec is not None else _delta_query_from_target(target_vec)
        candidates: list[DecodedCandidate] = []
        candidates.extend(self._decode_pair_targets(table_row, target_vec, delta_vec, prompt_vec, prompt_smiles, seed, config))
        if prompt_smiles:
            candidates.extend(self._decode_prompt_atom_edits(table_row, prompt_smiles, seed, config))
            candidates.extend(self._decode_prompt_fragments(table_row, delta_vec, prompt_smiles, seed, config))
        return _dedupe_rank(candidates, top_k=top_k)

    def _decode_pair_targets(
        self,
        table_row: dict[str, float],
        target_vec: np.ndarray,
        delta_vec: np.ndarray,
        prompt_vec: np.ndarray | None,
        prompt_smiles: str | None,
        seed: int,
        config: MMPTransformConfig,
    ) -> list[DecodedCandidate]:
        if not self.pairs:
            return []
        indices: list[int] = []
        indices.extend(_nearest_indices(self.target_x / TABLE_SCALES, self.target_nn, target_vec / TABLE_SCALES, config.target_neighbors))
        indices.extend(_nearest_indices(self.delta_x / TABLE_SCALES, self.delta_nn, delta_vec / TABLE_SCALES, config.delta_neighbors))
        if prompt_vec is not None:
            indices.extend(_nearest_indices(self.source_x / TABLE_SCALES, self.source_nn, prompt_vec / TABLE_SCALES, config.source_neighbors))
        out = []
        for idx in list(dict.fromkeys(indices)):
            pair = self.pairs[int(idx)]
            score = _pair_score(table_row, target_vec, delta_vec, pair, prompt_smiles, seed, config, train_smiles=self.train_smiles)
            out.append(
                DecodedCandidate(
                    smiles=pair.target_smiles,
                    score=score,
                    valid=True,
                    descriptors=pair.target_descriptors,
                    source="mmp_transform_target_decoder",
                )
            )
        return out

    def _decode_prompt_fragments(
        self,
        table_row: dict[str, float],
        delta_vec: np.ndarray,
        prompt_smiles: str,
        seed: int,
        config: MMPTransformConfig,
    ) -> list[DecodedCandidate]:
        if not self.fragments:
            return []
        indices = _nearest_indices(
            self.fragment_delta_x / TABLE_SCALES,
            self.fragment_nn,
            delta_vec / TABLE_SCALES,
            min(config.fragment_neighbors, len(self.fragments)),
        )
        out = []
        for idx in indices:
            fragment = self.fragments[int(idx)]
            for smi in _attach_fragment_cached(prompt_smiles, fragment.fragment_smiles, config.attachment_limit):
                candidate = self._fragment_candidate(
                    table_row=table_row,
                    smiles=smi,
                    seed=seed,
                    config=config,
                    prompt_smiles=prompt_smiles,
                    fragment=fragment,
                    source="mmp_learned_fragment_grow_decoder",
                )
                if candidate is not None:
                    out.append(candidate)
        if config.fragment_growth_steps >= 2:
            out.extend(self._decode_second_step_fragments(table_row, out, prompt_smiles, seed, config))
        return out

    def _decode_prompt_atom_edits(
        self,
        table_row: dict[str, float],
        prompt_smiles: str,
        seed: int,
        config: MMPTransformConfig,
    ) -> list[DecodedCandidate]:
        out = []
        for smi in _direct_atom_edit_smiles_from_row(prompt_smiles, table_row, config.attachment_limit):
            rec = molecular_descriptors(smi)
            if not rec.valid:
                continue
            score = _candidate_score(table_row, rec.descriptors, rec.smiles, seed, config, prompt_smiles=prompt_smiles, train_smiles=self.train_smiles)
            score -= 0.5 * config.fragment_bonus
            if rec.smiles in self.train_smiles:
                score += config.fragment_exact_penalty
            out.append(
                DecodedCandidate(
                    smiles=rec.smiles,
                    score=score,
                    valid=True,
                    descriptors=rec.descriptors,
                    source="mmp_source_atom_edit_decoder",
                )
            )
        return out

    def _decode_second_step_fragments(
        self,
        table_row: dict[str, float],
        first_step: list[DecodedCandidate],
        prompt_smiles: str,
        seed: int,
        config: MMPTransformConfig,
    ) -> list[DecodedCandidate]:
        if not first_step or not self.fragments:
            return []
        target_mw = float(table_row.get("MW", 0.0))
        seeds = [
            candidate
            for candidate in sorted(first_step, key=lambda item: item.score)[: max(1, config.fragment_growth_beam_size)]
            if target_mw - float(candidate.descriptors.get("MW", 0.0)) >= config.fragment_growth_mw_gap
        ]
        out = []
        for beam_idx, base in enumerate(seeds):
            remaining_delta = _table_vector(table_row) - _descriptor_vector(base.descriptors)
            indices = _nearest_indices(
                self.fragment_delta_x / TABLE_SCALES,
                self.fragment_nn,
                remaining_delta / TABLE_SCALES,
                min(config.fragment_second_step_neighbors, len(self.fragments)),
            )
            for idx in indices:
                fragment = self.fragments[int(idx)]
                for smi in _attach_fragment_cached(base.smiles, fragment.fragment_smiles, max(1, config.attachment_limit - 1)):
                    candidate = self._fragment_candidate(
                        table_row=table_row,
                        smiles=smi,
                        seed=seed + 10007 + beam_idx,
                        config=config,
                        prompt_smiles=prompt_smiles,
                        fragment=fragment,
                        source="mmp_two_step_fragment_grow_decoder",
                    )
                    if candidate is not None:
                        candidate = DecodedCandidate(
                            smiles=candidate.smiles,
                            score=candidate.score + 0.03,
                            valid=candidate.valid,
                            descriptors=candidate.descriptors,
                            source=candidate.source,
                        )
                        out.append(candidate)
        return out

    def _fragment_candidate(
        self,
        table_row: dict[str, float],
        smiles: str,
        seed: int,
        config: MMPTransformConfig,
        prompt_smiles: str,
        fragment: MMPFragment,
        source: str,
    ) -> DecodedCandidate | None:
        rec = molecular_descriptors(smiles)
        if not rec.valid:
            return None
        score = _candidate_score(table_row, rec.descriptors, rec.smiles, seed, config, prompt_smiles=prompt_smiles, train_smiles=self.train_smiles)
        score -= config.fragment_bonus
        score -= min(0.10, 0.01 * np.log1p(fragment.frequency))
        if rec.smiles in self.train_smiles:
            score += config.fragment_exact_penalty
        return DecodedCandidate(
            smiles=rec.smiles,
            score=score,
            valid=True,
            descriptors=rec.descriptors,
            source=source,
        )


def decode_mmp_table_row(
    row: dict[str, float],
    top_k: int,
    seed: int,
    index: MMPTransformIndex | None,
    config: MMPTransformConfig | None = None,
    prompt_smiles: str | None = None,
) -> list[DecodedCandidate]:
    if index is None:
        return []
    return index.decode(row, top_k=top_k, seed=seed, config=config or MMPTransformConfig(), prompt_smiles=prompt_smiles)


def build_transform_library_dataframe(df: pd.DataFrame, config: MMPTransformConfig | None = None) -> pd.DataFrame:
    config = config or MMPTransformConfig()
    records = _records_from_dataframe(df)
    pairs = mine_mmp_pairs(records, config=config)
    fragments = mine_fragments(records, config=config)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        row: dict[str, Any] = {
            "record_type": "pair",
            "source_smiles": pair.source_smiles,
            "target_smiles": pair.target_smiles,
            "similarity": pair.similarity,
            "edit_tags": pair.edit_tags,
        }
        for idx, col in enumerate(TABLE_COLUMNS):
            row[f"source_{col}"] = float(pair.source_vec[idx])
            row[f"target_{col}"] = float(pair.target_vec[idx])
            row[f"delta_{col}"] = float(pair.delta_vec[idx])
        rows.append(row)
    for fragment in fragments:
        row = {
            "record_type": "fragment",
            "fragment_smiles": fragment.fragment_smiles,
            "heavy_atoms": fragment.heavy_atoms,
            "frequency": fragment.frequency,
        }
        for idx, col in enumerate(TABLE_COLUMNS):
            row[f"frag_delta_{col}"] = float(fragment.approx_delta[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def mine_mmp_pairs(records: list[dict[str, Any]], config: MMPTransformConfig) -> list[MMPPair]:
    if len(records) < 2:
        return []
    x = np.asarray([_descriptor_vector(item["descriptors"]) for item in records], dtype=np.float32)
    scaled = x / TABLE_SCALES
    n_neighbors = min(max(2, config.pairs_per_source + 1), len(records))
    if SKLEARN_NN_AVAILABLE:
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(scaled)
        _, neighbor_idx = nn.kneighbors(scaled)
    else:
        neighbor_idx = _fallback_neighbor_indices(scaled, n_neighbors)
    pairs: list[MMPPair] = []
    seen: set[tuple[str, str]] = set()
    for src_idx, neighbors in iter_progress(enumerate(neighbor_idx), total=len(neighbor_idx), label="mining MMP pairs"):
        source = records[int(src_idx)]
        source_vec = x[int(src_idx)]
        for target_idx in neighbors:
            target_idx = int(target_idx)
            if target_idx == src_idx:
                continue
            target = records[target_idx]
            key = (source["smiles"], target["smiles"])
            if key in seen:
                continue
            seen.add(key)
            sim = tanimoto(source["smiles"], target["smiles"])
            if sim < config.min_pair_similarity or sim > config.max_pair_similarity:
                continue
            target_vec = x[target_idx]
            pairs.append(
                MMPPair(
                    source_smiles=source["smiles"],
                    target_smiles=target["smiles"],
                    source_vec=source_vec,
                    target_vec=target_vec,
                    delta_vec=target_vec - source_vec,
                    target_descriptors=target["descriptors"],
                    edit_tags=_edit_tags(source["descriptors"], target["descriptors"]),
                    similarity=float(sim),
                )
            )
            if len(pairs) >= config.max_pairs:
                return pairs
    return pairs


def mine_fragments(records: list[dict[str, Any]], config: MMPTransformConfig) -> list[MMPFragment]:
    if not chem_mod.RDKIT_AVAILABLE:
        return []
    counts: Counter[str] = Counter()
    approx: dict[str, np.ndarray] = {}
    heavy: dict[str, int] = {}
    for record in iter_progress(records, total=len(records), label="mining attachable fragments"):
        if len(counts) >= config.max_fragments * 3:
            break
        fragments = []
        fragments.extend(_extract_attachable_fragments(record["smiles"], config.max_fragment_atoms))
        fragments.extend(_extract_brics_fragments(record["smiles"], config.max_fragment_atoms))
        for fragment_smiles, delta, n_heavy in fragments:
            counts[fragment_smiles] += 1
            approx[fragment_smiles] = delta
            heavy[fragment_smiles] = n_heavy
    fragments = [
        MMPFragment(fragment_smiles=smi, approx_delta=approx[smi], heavy_atoms=heavy[smi], frequency=freq)
        for smi, freq in counts.most_common(config.max_fragments)
    ]
    return fragments


def _records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    seen = set()
    source = df.dropna(subset=["smiles"])
    for _, row in iter_progress(source.iterrows(), total=len(source), label="loading molecule descriptors"):
        can = canonicalize_smiles(str(row["smiles"]))
        if can is None or can in seen:
            continue
        descriptors = {col: float(row[col]) for col in TABLE_COLUMNS if col in row and not pd.isna(row[col])}
        if len(descriptors) < len(TABLE_COLUMNS):
            rec = molecular_descriptors(can)
            if not rec.valid:
                continue
            can = rec.smiles
            descriptors.update(rec.descriptors)
        if can in seen:
            continue
        seen.add(can)
        records.append({"smiles": can, "descriptors": {col: float(descriptors.get(col, 0.0)) for col in TABLE_COLUMNS}})
    return records


def _extract_attachable_fragments(smiles: str, max_fragment_atoms: int) -> list[tuple[str, np.ndarray, int]]:
    mol = chem_mod.Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out = []
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != chem_mod.Chem.BondType.SINGLE:
            continue
        if bond.GetBeginAtom().GetAtomicNum() == 1 or bond.GetEndAtom().GetAtomicNum() == 1:
            continue
        try:
            fragmented = chem_mod.Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(1, 1)])
            for frag in chem_mod.Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=False):
                normalized = _normalize_attachable_fragment(frag, max_fragment_atoms)
                if normalized is not None:
                    out.append(normalized)
        except Exception:
            continue
    return out


def _extract_brics_fragments(smiles: str, max_fragment_atoms: int) -> list[tuple[str, np.ndarray, int]]:
    """Mine BRICS leaf fragments with a single attachment dummy.

    The single-bond cutter above can be too brittle on server RDKit builds when
    sanitized dummy fragments fail. BRICS gives chemically meaningful leaves and
    usually produces fragments such as ``[16*]c1ccccc1`` that our attachment
    code can graft onto a prompt scaffold.
    """

    try:
        from rdkit.Chem import BRICS
    except Exception:  # pragma: no cover - depends on RDKit extras.
        return []
    mol = chem_mod.Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out = []
    try:
        fragments = BRICS.BRICSDecompose(mol, returnMols=True)
    except Exception:
        return []
    for frag in fragments:
        normalized = _normalize_attachable_fragment(frag, max_fragment_atoms)
        if normalized is not None:
            out.append(normalized)
    return out


def _normalize_attachable_fragment(fragment_mol, max_fragment_atoms: int) -> tuple[str, np.ndarray, int] | None:
    dummy_atoms = [atom for atom in fragment_mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        return None
    dummy_neighbors = list(dummy_atoms[0].GetNeighbors())
    if len(dummy_neighbors) != 1:
        return None
    heavy_atoms = sum(1 for atom in fragment_mol.GetAtoms() if atom.GetAtomicNum() > 1)
    if heavy_atoms < 1 or heavy_atoms > max_fragment_atoms:
        return None
    try:
        smi = chem_mod.Chem.MolToSmiles(fragment_mol, canonical=True)
        parsed = chem_mod.Chem.MolFromSmiles(smi, sanitize=False)
        if parsed is None:
            return None
        return smi, _fragment_approx_delta(fragment_mol), heavy_atoms
    except Exception:
        return None


def _fragment_approx_delta(fragment_mol) -> np.ndarray:
    counts = Counter(atom.GetSymbol() for atom in fragment_mol.GetAtoms() if atom.GetAtomicNum() > 0)
    vec = np.zeros(len(TABLE_COLUMNS), dtype=np.float32)
    for idx, col in enumerate(TABLE_COLUMNS):
        if col == "MW":
            vec[idx] = sum(counts.get(atom, 0) * weight for atom, weight in chem_mod.ATOM_WEIGHTS.items())
        elif col == "LogP":
            vec[idx] = 0.35 * counts.get("C", 0) + 0.25 * (counts.get("F", 0) + counts.get("Cl", 0) + counts.get("Br", 0)) - 0.30 * (
                counts.get("N", 0) + counts.get("O", 0) + counts.get("S", 0)
            )
        elif col == "TPSA":
            vec[idx] = 12.0 * counts.get("N", 0) + 17.0 * counts.get("O", 0) + 25.0 * counts.get("S", 0)
        elif col == "HBD":
            vec[idx] = min(2.0, counts.get("N", 0) + counts.get("O", 0))
        elif col == "HBA":
            vec[idx] = min(3.0, counts.get("N", 0) + counts.get("O", 0) + counts.get("S", 0))
        elif col == "RB":
            vec[idx] = max(0.0, sum(counts.values()) - 2.0)
        elif col in counts:
            vec[idx] = float(counts[col])
        elif col == "fg_halogen":
            vec[idx] = float(counts.get("F", 0) + counts.get("Cl", 0) + counts.get("Br", 0) + counts.get("I", 0))
        elif col == "fg_amine":
            vec[idx] = float(counts.get("N", 0))
        elif col == "fg_alcohol":
            vec[idx] = float(counts.get("O", 0))
    vec[TABLE_COLUMNS.index("SA")] = 0.05 * float(sum(counts.values()))
    return vec


@lru_cache(maxsize=200000)
def _direct_atom_edit_smiles(prompt_smiles: str, table_key: tuple[tuple[str, float], ...] | dict[str, float], attachment_limit: int) -> tuple[str, ...]:
    if not chem_mod.RDKIT_AVAILABLE:
        return tuple()
    table_row = dict(table_key) if not isinstance(table_key, dict) else table_key
    mol = chem_mod.Chem.MolFromSmiles(prompt_smiles)
    if mol is None:
        return tuple()
    out: set[str] = set()
    source_rec = molecular_descriptors(prompt_smiles)
    source_desc = source_rec.descriptors if source_rec.valid else {}

    halogen_delta = float(table_row.get("fg_halogen", 0.0)) - float(source_desc.get("fg_halogen", 0.0))
    hetero_delta = (
        float(table_row.get("N", 0.0) + table_row.get("O", 0.0) + table_row.get("S", 0.0))
        - float(source_desc.get("N", 0.0) + source_desc.get("O", 0.0) + source_desc.get("S", 0.0))
    )
    if halogen_delta >= 0.5:
        for symbol in ("F", "Cl"):
            out.update(_attach_atom_to_prompt(mol, symbol, attachment_limit))
    elif halogen_delta <= -0.5:
        out.update(_remove_terminal_atoms(mol, {9, 17, 35, 53}, attachment_limit))

    if hetero_delta >= 0.5:
        if float(table_row.get("O", 0.0)) >= float(source_desc.get("O", 0.0)) + 0.5:
            out.update(_attach_atom_to_prompt(mol, "O", attachment_limit))
        if float(table_row.get("N", 0.0)) >= float(source_desc.get("N", 0.0)) + 0.5:
            out.update(_attach_atom_to_prompt(mol, "N", attachment_limit))
    elif hetero_delta <= -0.5:
        out.update(_remove_terminal_atoms(mol, {7, 8, 16}, attachment_limit))

    return tuple(sorted(out))


def _direct_atom_edit_smiles_from_row(prompt_smiles: str, table_row: dict[str, float], attachment_limit: int) -> tuple[str, ...]:
    key = tuple(sorted((str(k), float(v)) for k, v in table_row.items()))
    return _direct_atom_edit_smiles(prompt_smiles, key, attachment_limit)


def _attach_atom_to_prompt(mol, symbol: str, attachment_limit: int) -> set[str]:
    out = set()
    mol_h = chem_mod.Chem.AddHs(mol)
    for prompt_idx in _attachment_atoms(mol)[: max(1, attachment_limit)]:
        prompt_atom = mol_h.GetAtomWithIdx(int(prompt_idx))
        hydrogen_neighbors = [nbr.GetIdx() for nbr in prompt_atom.GetNeighbors() if nbr.GetAtomicNum() == 1]
        if not hydrogen_neighbors:
            continue
        rw = chem_mod.Chem.RWMol(mol_h)
        try:
            rw.RemoveAtom(int(hydrogen_neighbors[0]))
            atom_idx = rw.AddAtom(chem_mod.Chem.Atom(symbol))
            rw.AddBond(int(prompt_idx), int(atom_idx), chem_mod.Chem.BondType.SINGLE)
            candidate = rw.GetMol()
            chem_mod.Chem.SanitizeMol(candidate)
            candidate = chem_mod.Chem.RemoveHs(candidate)
            out.add(chem_mod.Chem.MolToSmiles(candidate, canonical=True))
        except Exception:
            continue
    return out


def _remove_terminal_atoms(mol, atomic_nums: set[int], limit: int) -> set[str]:
    out = set()
    removable = [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() in atomic_nums and len([nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]) <= 1
    ][: max(1, limit)]
    for atom_idx in removable:
        rw = chem_mod.Chem.RWMol(mol)
        try:
            rw.RemoveAtom(int(atom_idx))
            candidate = rw.GetMol()
            chem_mod.Chem.SanitizeMol(candidate)
            smi = chem_mod.Chem.MolToSmiles(candidate, canonical=True)
            if smi:
                out.add(smi)
        except Exception:
            continue
    return out


@lru_cache(maxsize=300000)
def _attach_fragment_cached(prompt_smiles: str, fragment_smiles: str, attachment_limit: int) -> tuple[str, ...]:
    if not chem_mod.RDKIT_AVAILABLE:
        return tuple()
    prompt = chem_mod.Chem.MolFromSmiles(prompt_smiles)
    fragment = chem_mod.Chem.MolFromSmiles(fragment_smiles)
    if prompt is None or fragment is None:
        return tuple()
    dummy_atoms = [atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        return tuple()
    dummy_idx = dummy_atoms[0]
    dummy_neighbors = [atom.GetIdx() for atom in fragment.GetAtomWithIdx(dummy_idx).GetNeighbors()]
    if len(dummy_neighbors) != 1:
        return tuple()
    frag_attach_idx = dummy_neighbors[0]
    prompt_atoms = _attachment_atoms(prompt)[: max(1, attachment_limit)]
    out = set()
    for prompt_idx in prompt_atoms:
        candidate = _combine_prompt_fragment(prompt, fragment, prompt_idx, dummy_idx, frag_attach_idx)
        if candidate:
            out.add(candidate)
    return tuple(sorted(out))


def _combine_prompt_fragment(prompt, fragment, prompt_idx: int, dummy_idx: int, frag_attach_idx: int) -> str | None:
    rw = chem_mod.Chem.RWMol(prompt)
    mapping: dict[int, int] = {}
    for atom in fragment.GetAtoms():
        idx = int(atom.GetIdx())
        if idx == dummy_idx:
            continue
        new_atom = chem_mod.Chem.Atom(atom)
        mapping[idx] = rw.AddAtom(new_atom)
    try:
        for bond in fragment.GetBonds():
            a = int(bond.GetBeginAtomIdx())
            b = int(bond.GetEndAtomIdx())
            if dummy_idx in {a, b}:
                continue
            rw.AddBond(mapping[a], mapping[b], bond.GetBondType())
        rw.AddBond(prompt_idx, mapping[frag_attach_idx], chem_mod.Chem.BondType.SINGLE)
        mol = rw.GetMol()
        chem_mod.Chem.SanitizeMol(mol)
        return chem_mod.Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _attachment_atoms(mol) -> list[int]:
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in {1, 9, 17, 35, 53}:
            continue
        if atom.GetTotalNumHs() > 0:
            atoms.append(int(atom.GetIdx()))
    return atoms


def _pair_score(
    table_row: dict[str, float],
    target_vec: np.ndarray,
    delta_vec: np.ndarray,
    pair: MMPPair,
    prompt_smiles: str | None,
    seed: int,
    config: MMPTransformConfig,
    train_smiles: set[str],
) -> float:
    score = 0.62 * float(np.mean(np.abs((target_vec - pair.target_vec) / TABLE_SCALES)))
    score += 0.28 * float(np.mean(np.abs((delta_vec - pair.delta_vec) / TABLE_SCALES)))
    score -= config.transform_bonus
    score -= 0.10 * min(1.0, pair.similarity)
    score += _candidate_score(table_row, pair.target_descriptors, pair.target_smiles, seed, config, prompt_smiles=prompt_smiles, train_smiles=train_smiles)
    if pair.target_smiles in set([pair.source_smiles]):
        score += 3.0
    return float(score)


def _candidate_score(
    table_row: dict[str, float],
    descriptors: dict[str, float],
    smiles: str,
    seed: int,
    config: MMPTransformConfig,
    prompt_smiles: str | None = None,
    train_smiles: set[str] | None = None,
) -> float:
    target = np.asarray([float(table_row.get(col, 0.0)) for col in TARGET_COLUMNS], dtype=np.float32)
    actual = np.asarray([float(descriptors.get(col, 0.0)) for col in TARGET_COLUMNS], dtype=np.float32)
    score = float(np.mean(np.abs(target - actual) / TARGET_SCALES))
    count_target = np.asarray([float(table_row.get(col, 0.0)) for col in TABLE_COLUMNS[8:]], dtype=np.float32)
    count_actual = np.asarray([float(descriptors.get(col, 0.0)) for col in TABLE_COLUMNS[8:]], dtype=np.float32)
    score += 0.12 * float(np.mean(np.abs(count_target - count_actual) / COUNT_SCALES))
    if not passes_druglike_filters(descriptors):
        score += DRUGLIKE_SOFT_PENALTY
    if train_smiles and smiles in train_smiles:
        score += config.exact_train_penalty
    if prompt_smiles:
        if _contains_prompt(prompt_smiles, smiles):
            score -= config.prompt_match_bonus
        else:
            score += config.prompt_miss_penalty
    score += 0.012 * _stable_noise(smiles, seed)
    return float(score)

def _delta_query_from_target(target_vec: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(target_vec)
    delta[: len(TARGET_COLUMNS)] = target_vec[: len(TARGET_COLUMNS)] - np.asarray([350.0, 2.5, 0.65, 70.0, 1.0, 4.0, 5.0, 2.5], dtype=np.float32)
    delta[len(TARGET_COLUMNS) :] = 0.0
    return delta


def _edit_tags(source: dict[str, float], target: dict[str, float]) -> str:
    tags = []
    for prop, threshold in [("MW", 40.0), ("LogP", 0.5), ("QED", 0.08), ("TPSA", 15.0), ("HBA", 1.0), ("HBD", 1.0), ("RB", 1.0)]:
        delta = float(target.get(prop, 0.0) - source.get(prop, 0.0))
        if delta >= threshold:
            tags.append(f"increase_{prop}")
        elif delta <= -threshold:
            tags.append(f"decrease_{prop}")
    if float(target.get("fg_halogen", 0.0) - source.get("fg_halogen", 0.0)) >= 1:
        tags.append("add_halogen")
    if float(target.get("fg_amide", 0.0) - source.get("fg_amide", 0.0)) >= 1:
        tags.append("add_amide")
    return "|".join(tags)


def _descriptor_vector(descriptors: dict[str, float]) -> np.ndarray:
    return np.asarray([float(descriptors.get(col, 0.0)) for col in TABLE_COLUMNS], dtype=np.float32)


def _table_vector(table_row: dict[str, float]) -> np.ndarray:
    return np.asarray([float(table_row.get(col, 0.0)) for col in TABLE_COLUMNS], dtype=np.float32)


def _smiles_vector(smiles: str | None) -> np.ndarray | None:
    if not smiles:
        return None
    rec = molecular_descriptors(smiles)
    if not rec.valid:
        return None
    return _descriptor_vector(rec.descriptors)


def _fit_nn(matrix: np.ndarray):
    if not SKLEARN_NN_AVAILABLE or len(matrix) == 0:
        return None
    return NearestNeighbors(n_neighbors=min(512, len(matrix)), metric="euclidean").fit(matrix)


def _nearest_indices(matrix: np.ndarray, nn: Any, scaled_query: np.ndarray, n: int) -> list[int]:
    if n <= 0 or len(matrix) == 0:
        return []
    k = min(int(n), len(matrix))
    if nn is not None:
        _, indices = nn.kneighbors(scaled_query[None, :], n_neighbors=k)
        return [int(idx) for idx in indices[0]]
    distances = np.mean(np.abs(matrix - scaled_query), axis=1)
    chosen = np.argpartition(distances, k - 1)[:k]
    return [int(idx) for idx in chosen[np.argsort(distances[chosen])]]


def _fallback_neighbor_indices(scaled: np.ndarray, n_neighbors: int) -> np.ndarray:
    rows = []
    for idx, row in enumerate(scaled):
        distances = np.mean(np.abs(scaled - row), axis=1)
        chosen = np.argpartition(distances, n_neighbors - 1)[:n_neighbors]
        rows.append(chosen[np.argsort(distances[chosen])])
    return np.asarray(rows, dtype=int)


def _dedupe_rank(candidates: list[DecodedCandidate], top_k: int) -> list[DecodedCandidate]:
    by_smiles: dict[str, DecodedCandidate] = {}
    for candidate in candidates:
        existing = by_smiles.get(candidate.smiles)
        if existing is None or candidate.score < existing.score:
            by_smiles[candidate.smiles] = candidate
    ranked = list(by_smiles.values())
    ranked.sort(key=lambda item: item.score)
    return ranked[:top_k]


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
            return bool(prompt_mol is not None and candidate_mol is not None and candidate_mol.HasSubstructMatch(prompt_mol))
        except Exception:
            return False
    return prompt in candidate or tanimoto(prompt, candidate) >= 0.5


def _stable_noise(smiles: str, seed: int) -> float:
    digest = hashlib.sha256(f"mmp-transform-decoder:{seed}:{smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(default) if np.isnan(out) else out
