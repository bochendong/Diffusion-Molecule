"""Latent-to-molecule decoding baselines."""

from __future__ import annotations

import numpy as np

from .chem import canonicalize_smiles, molecular_descriptors, scaffold_key
from .property_guidance import parse_property_targets, property_match_score
from .schema import BenchmarkExample, Candidate, TaskType

class RetrievalDecoder:
    """Nearest-neighbor molecular latent decoder.

    This is intentionally a baseline. It gives the JEPA planner a concrete
    molecule surface while future learned graph/SELFIES decoders are built.
    """

    def __init__(
        self,
        smiles: list[str],
        latents: np.ndarray,
        de_novo_latent_rerank_weight: float = 0.05,
        source_rerank_weight: float = 0.35,
        property_rerank_weight: float = 0.25,
        scaffold_rerank_bonus: float = 0.15,
    ):
        self.smiles = [canonicalize_smiles(smi) or smi for smi in smiles]
        self.latents = np.asarray(latents, dtype=np.float32)
        self.descriptors = [molecular_descriptors(smi).descriptors for smi in self.smiles]
        self.scaffolds = [scaffold_key(smi) for smi in self.smiles]
        self.de_novo_latent_rerank_weight = float(de_novo_latent_rerank_weight)
        self.source_rerank_weight = float(source_rerank_weight)
        self.property_rerank_weight = float(property_rerank_weight)
        self.scaffold_rerank_bonus = float(scaffold_rerank_bonus)

    def decode(
        self,
        pred_latents: np.ndarray,
        source_smiles: list[str | None],
        top_k: int = 5,
        examples: list[BenchmarkExample] | None = None,
        source_latents: np.ndarray | None = None,
    ) -> list[list[Candidate]]:
        pred_latents = np.asarray(pred_latents, dtype=np.float32)
        sims = _cosine_similarity(pred_latents, self.latents)
        source_sims = _cosine_similarity(np.asarray(source_latents, dtype=np.float32), self.latents) if source_latents is not None else None
        all_candidates: list[list[Candidate]] = []
        for row_idx in range(len(pred_latents)):
            example = examples[row_idx] if examples else None
            row_source_sims = source_sims[row_idx] if source_sims is not None else None
            order, origin, candidate_scores = self._candidate_order(sims[row_idx], row_source_sims, example)
            seen: set[str] = set()
            row: list[Candidate] = []
            source = canonicalize_smiles(source_smiles[row_idx]) if source_smiles[row_idx] else None
            if source:
                seen.add(source)
            for idx in order:
                smiles = self.smiles[int(idx)]
                if smiles in seen:
                    continue
                row.append(Candidate(smiles=smiles, origin=origin, score=float(candidate_scores[int(idx)]), rank=len(row) + 1))
                seen.add(smiles)
                if len(row) >= top_k:
                    break
            all_candidates.append(row)
        return all_candidates

    def _candidate_order(
        self,
        latent_sims: np.ndarray,
        source_sims: np.ndarray | None,
        example: BenchmarkExample | None,
    ) -> tuple[np.ndarray, str, np.ndarray]:
        scores = np.asarray(latent_sims, dtype=np.float32).copy()
        if example and example.task_type == TaskType.DE_NOVO:
            targets = parse_property_targets(example.instruction)
            if targets:
                scores = np.asarray([property_match_score(desc, targets) for desc in self.descriptors], dtype=np.float32)
                scores = scores + self.de_novo_latent_rerank_weight * latent_sims
                return np.argsort(-scores), "property_guided_retrieval", scores
            return np.argsort(-scores), "latent_retrieval", scores
        if example and source_sims is not None and example.task_type in {TaskType.EDIT, TaskType.INPAINT, TaskType.FRAGMENT_GROW}:
            scores = scores + self.source_rerank_weight * source_sims
            targets = parse_property_targets(example.instruction)
            if targets:
                property_scores = np.asarray([property_match_score(desc, targets) for desc in self.descriptors], dtype=np.float32)
                scores = scores + self.property_rerank_weight * property_scores
            source_scaffold = scaffold_key(example.source_smiles)
            if source_scaffold and _asks_to_preserve_source(example):
                scaffold_scores = np.asarray([1.0 if scaffold == source_scaffold else 0.0 for scaffold in self.scaffolds], dtype=np.float32)
                scores = scores + self.scaffold_rerank_bonus * scaffold_scores
            return np.argsort(-scores), "task_guided_retrieval", scores
        return np.argsort(-scores), "latent_retrieval", scores


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).T
    denom = np.maximum(left_norm * right_norm, 1e-8)
    return (left @ right.T) / denom


def _asks_to_preserve_source(example: BenchmarkExample) -> bool:
    text = f"{example.instruction} {example.mask_hint or ''} {' '.join(example.goals)}".lower()
    return any(token in text for token in ("preserve", "core", "fragment", "scaffold", "recognizable"))
