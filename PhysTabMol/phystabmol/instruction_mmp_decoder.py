"""Matched-pair transformation retrieval for instruction-guided editing.

This is a pragmatic MMP-inspired decoder. It does not try to invent arbitrary
reaction chemistry. Instead, it learns from verified training pairs:

    train_source -> train_target under train_instruction

At decode time it retrieves transformations whose source molecule, descriptor
delta, and instruction tags resemble the current request, then re-ranks the
resulting target molecules with the deterministic verifier for the current
source and instruction.
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
from .instruction_schema import TAG_NAMES, normalize_spec
from .instruction_verifier import verify_instruction
from .schema import TABLE_COLUMNS


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
class MMPDecoderConfig:
    pool_size: int = 768
    source_neighbors: int = 256
    delta_neighbors: int = 256
    tag_neighbors: int = 384
    reference_neighbors: int = 128
    verify_candidates: int = 512
    source_copy_penalty: float = 10.0
    failed_goal_penalty: float = 4.0
    failed_constraint_penalty: float = 6.0
    failed_edit_penalty: float = 3.0
    reference_similarity_bonus: float = 1.5
    train_source_similarity_bonus: float = 1.25


@dataclass(frozen=True)
class MMPTransformation:
    source_smiles: str
    target_smiles: str
    spec_json: str
    source_vec: np.ndarray
    target_vec: np.ndarray
    delta_vec: np.ndarray
    tag_vec: np.ndarray
    target_descriptors: dict[str, float]


class MMPTransformationIndex:
    def __init__(self, transforms: list[MMPTransformation]):
        if not transforms:
            raise ValueError("MMP decoder needs at least one verified train transformation.")
        self.transforms = transforms
        self.source_x = np.asarray([item.source_vec for item in transforms], dtype=np.float32)
        self.target_x = np.asarray([item.target_vec for item in transforms], dtype=np.float32)
        self.delta_x = np.asarray([item.delta_vec for item in transforms], dtype=np.float32)
        self.tag_x = np.asarray([item.tag_vec for item in transforms], dtype=np.float32)
        self.scaled_source_x = self.source_x / TABLE_SCALES
        self.scaled_target_x = self.target_x / TABLE_SCALES
        self.scaled_delta_x = self.delta_x / TABLE_SCALES
        self.source_nn = None
        self.delta_nn = None
        self.target_nn = None
        if SKLEARN_NN_AVAILABLE:
            n_neighbors = min(len(self.transforms), 512)
            self.source_nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(self.scaled_source_x)
            self.delta_nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(self.scaled_delta_x)
            self.target_nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(self.scaled_target_x)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "MMPTransformationIndex":
        required = {"source_smiles", "target_smiles", "instruction_spec_json"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for MMP decoder: {sorted(missing)}")

        transforms = []
        seen = set()
        rows = df.dropna(subset=["source_smiles", "target_smiles", "instruction_spec_json"])
        for _, row in rows.iterrows():
            source_can = canonicalize_smiles(str(row["source_smiles"]))
            target_can = canonicalize_smiles(str(row["target_smiles"]))
            if source_can is None or target_can is None or source_can == target_can:
                continue
            spec_json = str(row["instruction_spec_json"])
            key = (source_can, target_can, spec_json)
            if key in seen:
                continue
            seen.add(key)
            source = molecular_descriptors(source_can)
            target = molecular_descriptors(target_can)
            if not source.valid or not target.valid:
                continue
            transforms.append(
                MMPTransformation(
                    source_smiles=source.smiles,
                    target_smiles=target.smiles,
                    spec_json=spec_json,
                    source_vec=_descriptor_vector(source.descriptors),
                    target_vec=_descriptor_vector(target.descriptors),
                    delta_vec=_descriptor_vector(target.descriptors) - _descriptor_vector(source.descriptors),
                    tag_vec=_tag_vector(spec_json),
                    target_descriptors=target.descriptors,
                )
            )
        return cls(transforms)

    def decode(
        self,
        table_row: dict[str, float],
        instruction_row: Any,
        top_k: int,
        seed: int,
        config: MMPDecoderConfig,
    ) -> list[DecodedCandidate]:
        source_smiles = str(instruction_row["source_smiles"])
        spec_json = str(instruction_row["instruction_spec_json"])
        reference_smiles = _optional_row_value(instruction_row, "reference_smiles")
        source_vec = _smiles_vector(source_smiles)
        if source_vec is None:
            return []
        plan_vec = _table_vector(table_row)
        plan_delta = plan_vec - source_vec
        query_tags = _tag_vector(spec_json)
        candidate_indices = self._candidate_indices(
            source_vec=source_vec,
            plan_delta=plan_delta,
            query_tags=query_tags,
            reference_smiles=reference_smiles,
            config=config,
        )

        ranked = []
        source_can = canonicalize_smiles(source_smiles) or source_smiles
        for idx in candidate_indices[: config.verify_candidates]:
            transform = self.transforms[int(idx)]
            candidate_smiles = transform.target_smiles
            result = _cached_verify(source_smiles, candidate_smiles, spec_json)
            score = _mmp_score(
                table_row=table_row,
                transform=transform,
                source_vec=source_vec,
                plan_delta=plan_delta,
                query_tags=query_tags,
                verification=result,
                source_smiles=source_smiles,
                reference_smiles=reference_smiles,
                spec_json=spec_json,
                seed=seed,
                config=config,
            )
            candidate = DecodedCandidate(
                smiles=candidate_smiles,
                score=score,
                valid=True,
                descriptors=transform.target_descriptors,
                source="mmp_transformation_retrieval_decoder",
            )
            ranked.append((_verification_rank_key(result, candidate, source_can, spec_json, score), candidate))

        ranked.sort(key=lambda item: item[0])
        deduped = []
        seen = set()
        for _, candidate in ranked:
            if candidate.smiles in seen:
                continue
            deduped.append(candidate)
            seen.add(candidate.smiles)
            if len(deduped) >= top_k:
                break
        return deduped

    def _candidate_indices(
        self,
        source_vec: np.ndarray,
        plan_delta: np.ndarray,
        query_tags: np.ndarray,
        reference_smiles: str | None,
        config: MMPDecoderConfig,
    ) -> list[int]:
        indices: list[int] = []
        indices.extend(self._nearest(self.scaled_source_x, self.source_nn, source_vec, config.source_neighbors))
        indices.extend(self._nearest(self.scaled_delta_x, self.delta_nn, plan_delta, config.delta_neighbors))
        indices.extend(self._tag_neighbors(query_tags, config.tag_neighbors))
        if reference_smiles:
            ref_vec = _smiles_vector(reference_smiles)
            if ref_vec is not None:
                indices.extend(self._nearest(self.scaled_target_x, self.target_nn, ref_vec, config.reference_neighbors))
        return list(dict.fromkeys(indices))[: config.pool_size]

    def _nearest(self, matrix: np.ndarray, nn: Any, raw_vec: np.ndarray, n: int) -> list[int]:
        if n <= 0:
            return []
        k = min(n, len(self.transforms))
        scaled = raw_vec / TABLE_SCALES
        if nn is not None:
            _, indices = nn.kneighbors(scaled[None, :], n_neighbors=k)
            return [int(idx) for idx in indices[0]]
        distances = np.mean(np.abs(matrix - scaled), axis=1)
        chosen = np.argpartition(distances, k - 1)[:k]
        return [int(idx) for idx in chosen[np.argsort(distances[chosen])]]

    def _tag_neighbors(self, query_tags: np.ndarray, n: int) -> list[int]:
        if n <= 0 or float(query_tags.sum()) <= 0.0:
            return []
        overlap = self.tag_x @ query_tags
        union = np.maximum(1.0, self.tag_x.sum(axis=1) + query_tags.sum() - overlap)
        scores = overlap / union
        k = min(n, len(self.transforms))
        chosen = np.argpartition(-scores, k - 1)[:k]
        return [int(idx) for idx in chosen[np.argsort(-scores[chosen])]]


def decode_mmp_transform(
    table_row: dict[str, float],
    instruction_row: Any,
    index: MMPTransformationIndex,
    top_k: int = 5,
    seed: int = 0,
    config: MMPDecoderConfig | None = None,
) -> list[DecodedCandidate]:
    return index.decode(
        table_row=table_row,
        instruction_row=instruction_row,
        top_k=top_k,
        seed=seed,
        config=config or MMPDecoderConfig(),
    )


def _mmp_score(
    table_row: dict[str, float],
    transform: MMPTransformation,
    source_vec: np.ndarray,
    plan_delta: np.ndarray,
    query_tags: np.ndarray,
    verification: dict[str, Any],
    source_smiles: str,
    reference_smiles: str | None,
    spec_json: str,
    seed: int,
    config: MMPDecoderConfig,
) -> float:
    plan_vec = _table_vector(table_row)
    target_distance = float(np.mean(np.abs(plan_vec - transform.target_vec) / TABLE_SCALES))
    delta_distance = float(np.mean(np.abs(plan_delta - transform.delta_vec) / TABLE_SCALES))
    source_distance = float(np.mean(np.abs(source_vec - transform.source_vec) / TABLE_SCALES))
    tag_overlap = _tag_overlap(query_tags, transform.tag_vec)

    score = 0.45 * target_distance + 0.35 * delta_distance + 0.20 * source_distance
    score -= 2.0 * tag_overlap
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
    score -= config.train_source_similarity_bonus * tanimoto(source_smiles, transform.source_smiles)
    if reference_smiles:
        score -= config.reference_similarity_bonus * tanimoto(transform.target_smiles, reference_smiles)
    source_can = canonicalize_smiles(source_smiles) or source_smiles
    if transform.target_smiles == source_can:
        score += config.source_copy_penalty
    normalized = normalize_spec(spec_json)
    if not passes_druglike_filters(transform.target_descriptors):
        score += DRUGLIKE_SOFT_PENALTY
        if "keep_druglike" in normalized["constraints"]:
            score += 2.0 * DRUGLIKE_SOFT_PENALTY
    score += 0.015 * _stable_noise(transform.target_smiles, seed)
    return float(score)


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


def _tag_vector(spec_json: str) -> np.ndarray:
    normalized = normalize_spec(spec_json)
    active = set(normalized["goals"]) | set(normalized["constraints"]) | set(normalized["edits"])
    return np.asarray([1.0 if tag in active else 0.0 for tag in TAG_NAMES], dtype=np.float32)


def _tag_overlap(query: np.ndarray, candidate: np.ndarray) -> float:
    if float(query.sum()) <= 0.0:
        return 0.0
    overlap = float(query @ candidate)
    union = max(1.0, float(query.sum() + candidate.sum() - overlap))
    return overlap / union


def _verification_rank_key(
    verification: dict[str, Any],
    candidate: DecodedCandidate,
    source_smiles: str,
    spec_json: str,
    score: float,
) -> tuple[float, ...]:
    normalized = normalize_spec(spec_json)
    source_copy = candidate.smiles == source_smiles
    nontrivial_edit = bool(normalized["goals"] or normalized["edits"])
    return (
        0.0 if verification.get("overall_success") else 1.0,
        0.0 if verification.get("constraint_success") else 1.0,
        0.0 if verification.get("goal_success") else 1.0,
        0.0 if verification.get("edit_success") else 1.0,
        1.0 if source_copy and nontrivial_edit else 0.0,
        -float(verification.get("similarity_to_source", 0.0)),
        float(score),
    )


@lru_cache(maxsize=300000)
def _cached_verify(source_smiles: str, candidate_smiles: str, spec_json: str) -> dict[str, Any]:
    return verify_instruction(source_smiles, candidate_smiles, spec_json)


def _stable_noise(smiles: str, seed: int) -> float:
    digest = hashlib.sha256(f"mmp-transform:{seed}:{smiles}".encode("utf-8")).digest()
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
