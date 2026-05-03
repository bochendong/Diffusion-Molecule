"""Source-aware candidate retrieval for instruction-guided molecular editing.

The generic table decoder can produce valid molecules, but it is not anchored
to the source molecule. This module adds a retrieval/repair stage over the
training molecule pool, then ranks candidates with the deterministic verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from typing import Any

import numpy as np
import pandas as pd

if os.environ.get("PHYSTABMOL_DISABLE_SKLEARN_NN", "0") == "1":  # pragma: no cover - local smoke path.
    NearestNeighbors = None
    SKLEARN_NN_AVAILABLE = False
else:
    try:  # pragma: no cover - server path; fallback below covers minimal environments.
        from sklearn.neighbors import NearestNeighbors

        SKLEARN_NN_AVAILABLE = True
    except Exception:  # pragma: no cover
        NearestNeighbors = None
        SKLEARN_NN_AVAILABLE = False

from .chem import canonicalize_smiles, molecular_descriptors, passes_druglike_filters, tanimoto
from .decoder import DRUGLIKE_SOFT_PENALTY, DecodedCandidate
from .instruction_schema import normalize_spec
from .instruction_verifier import verify_instruction
from .schema import TABLE_COLUMNS, TARGET_COLUMNS


TABLE_SCALES = np.asarray(
    [
        120.0,
        3.0,
        0.35,
        60.0,
        3.0,
        5.0,
        5.0,
        2.5,
        8.0,
        3.0,
        3.0,
        2.0,
        3.0,
        2.0,
        2.0,
        1.0,
        3.0,
        4.0,
        3.0,
        3.0,
        3.0,
        4.0,
        3.0,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class SourceAwareDecoderConfig:
    pool_size: int = 256
    plan_neighbors: int = 128
    source_neighbors: int = 128
    reference_neighbors: int = 64
    verify_candidates: int = 192
    include_source_fallback: bool = True
    source_copy_penalty: float = 8.0
    failed_goal_penalty: float = 4.0
    failed_constraint_penalty: float = 6.0
    failed_edit_penalty: float = 3.0
    reference_similarity_bonus: float = 2.0


class SourceAwareCandidateIndex:
    def __init__(self, candidates: list[DecodedCandidate], table_x: np.ndarray):
        if not candidates:
            raise ValueError("Source-aware decoder needs at least one candidate molecule.")
        self.candidates = candidates
        self.table_x = np.asarray(table_x, dtype=np.float32)
        self.scaled_x = self.table_x / TABLE_SCALES
        self.nn = None
        if SKLEARN_NN_AVAILABLE:
            n_neighbors = min(len(self.candidates), 512)
            self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(self.scaled_x)
        self.by_smiles = {candidate.smiles: idx for idx, candidate in enumerate(self.candidates)}

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "SourceAwareCandidateIndex":
        smiles = []
        for col in ("target_smiles", "source_smiles", "reference_smiles"):
            if col in df.columns:
                smiles.extend(df[col].dropna().astype(str).tolist())
        by_smiles: dict[str, DecodedCandidate] = {}
        rows = []
        for raw in smiles:
            can = canonicalize_smiles(raw)
            if can is None or can in by_smiles:
                continue
            rec = molecular_descriptors(can)
            if not rec.valid:
                continue
            candidate = DecodedCandidate(
                smiles=rec.smiles,
                score=0.0,
                valid=True,
                descriptors=rec.descriptors,
                source="source_aware_train_pool",
            )
            by_smiles[rec.smiles] = candidate
            rows.append([float(rec.descriptors.get(col, 0.0)) for col in TABLE_COLUMNS])
        return cls(list(by_smiles.values()), np.asarray(rows, dtype=np.float32))

    def decode(
        self,
        table_row: dict[str, float],
        instruction_row: Any,
        top_k: int,
        seed: int,
        config: SourceAwareDecoderConfig,
    ) -> list[DecodedCandidate]:
        source_smiles = str(instruction_row["source_smiles"])
        spec_json = str(instruction_row["instruction_spec_json"])
        reference_smiles = _optional_row_value(instruction_row, "reference_smiles")
        candidate_indices = self._candidate_indices(
            table_row=table_row,
            source_smiles=source_smiles,
            reference_smiles=reference_smiles,
            config=config,
        )
        ranked = []
        for idx in candidate_indices[: config.verify_candidates]:
            candidate = self.candidates[int(idx)]
            if candidate.smiles == source_smiles and not config.include_source_fallback:
                continue
            result = _cached_verify(source_smiles, candidate.smiles, spec_json)
            score = _source_aware_score(
                table_row=table_row,
                candidate=candidate,
                verification=result,
                seed=seed,
                source_smiles=source_smiles,
                reference_smiles=reference_smiles,
                spec_json=spec_json,
                config=config,
            )
            ranked.append(
                DecodedCandidate(
                    smiles=candidate.smiles,
                    score=score,
                    valid=candidate.valid,
                    descriptors=candidate.descriptors,
                    source="source_aware_retrieval_decoder",
                )
            )
        ranked.sort(key=lambda item: item.score)
        deduped = []
        seen = set()
        for candidate in ranked:
            if candidate.smiles in seen:
                continue
            deduped.append(candidate)
            seen.add(candidate.smiles)
            if len(deduped) >= top_k:
                break
        return deduped

    def _candidate_indices(
        self,
        table_row: dict[str, float],
        source_smiles: str,
        reference_smiles: str | None,
        config: SourceAwareDecoderConfig,
    ) -> list[int]:
        plan_vec = _table_vector(table_row)
        indices: list[int] = []
        indices.extend(self._nearest(plan_vec, config.plan_neighbors))
        source_idx = self.by_smiles.get(canonicalize_smiles(source_smiles) or source_smiles)
        if source_idx is not None:
            indices.extend(self._nearest(self.table_x[source_idx], config.source_neighbors))
            indices.append(source_idx)
        if reference_smiles:
            ref_idx = self.by_smiles.get(canonicalize_smiles(reference_smiles) or reference_smiles)
            if ref_idx is not None:
                indices.extend(self._nearest(self.table_x[ref_idx], config.reference_neighbors))
                indices.append(ref_idx)
        return list(dict.fromkeys(indices))[: config.pool_size]

    def _nearest(self, raw_vec: np.ndarray, n: int) -> list[int]:
        if n <= 0:
            return []
        k = min(n, len(self.candidates))
        scaled = raw_vec / TABLE_SCALES
        if self.nn is not None:
            _, indices = self.nn.kneighbors(scaled[None, :], n_neighbors=k)
            return [int(idx) for idx in indices[0]]
        distances = np.mean(np.abs(self.scaled_x - scaled), axis=1)
        chosen = np.argpartition(distances, k - 1)[:k]
        return [int(idx) for idx in chosen[np.argsort(distances[chosen])]]


def decode_source_aware(
    table_row: dict[str, float],
    instruction_row: Any,
    index: SourceAwareCandidateIndex,
    top_k: int = 5,
    seed: int = 0,
    config: SourceAwareDecoderConfig | None = None,
) -> list[DecodedCandidate]:
    return index.decode(
        table_row=table_row,
        instruction_row=instruction_row,
        top_k=top_k,
        seed=seed,
        config=config or SourceAwareDecoderConfig(),
    )


def _source_aware_score(
    table_row: dict[str, float],
    candidate: DecodedCandidate,
    verification: dict[str, Any],
    seed: int,
    source_smiles: str,
    reference_smiles: str | None,
    spec_json: str,
    config: SourceAwareDecoderConfig,
) -> float:
    plan_distance = float(np.mean(np.abs(_table_vector(table_row) - _candidate_vector(candidate)) / TABLE_SCALES))
    score = plan_distance
    score -= 30.0 if verification.get("overall_success") else 0.0
    if verification.get("constraint_success"):
        score -= 8.0
    else:
        score += config.failed_constraint_penalty
    if verification.get("goal_success"):
        score -= 6.0
    else:
        score += config.failed_goal_penalty
    if verification.get("edit_success"):
        score -= 4.0
    else:
        score += config.failed_edit_penalty
    score -= 0.8 * float(verification.get("similarity_to_source", 0.0))
    if reference_smiles:
        score -= config.reference_similarity_bonus * tanimoto(candidate.smiles, reference_smiles)
    source_can = canonicalize_smiles(source_smiles) or source_smiles
    if candidate.smiles == source_can:
        score += config.source_copy_penalty
    normalized = normalize_spec(spec_json)
    needs_druglike = "keep_druglike" in normalized["constraints"]
    if not passes_druglike_filters(candidate.descriptors):
        score += DRUGLIKE_SOFT_PENALTY
        if needs_druglike:
            score += 2.0 * DRUGLIKE_SOFT_PENALTY
    score += 0.015 * _stable_noise(candidate.smiles, seed)
    return float(score)


def _table_vector(table_row: dict[str, float]) -> np.ndarray:
    return np.asarray([float(table_row.get(col, 0.0)) for col in TABLE_COLUMNS], dtype=np.float32)


def _candidate_vector(candidate: DecodedCandidate) -> np.ndarray:
    return np.asarray([float(candidate.descriptors.get(col, 0.0)) for col in TABLE_COLUMNS], dtype=np.float32)


@lru_cache(maxsize=300000)
def _cached_verify(source_smiles: str, candidate_smiles: str, spec_json: str) -> dict[str, Any]:
    return verify_instruction(source_smiles, candidate_smiles, spec_json)


def _stable_noise(smiles: str, seed: int) -> float:
    digest = hashlib.sha256(f"source-aware:{seed}:{smiles}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _optional_row_value(row: Any, key: str) -> str | None:
    try:
        value = row[key]
    except Exception:
        value = getattr(row, key, None)
    if value is None:
        return None
    try:
        if bool(np.isnan(value)):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None
