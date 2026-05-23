"""Latent-to-molecule decoding baselines."""

from __future__ import annotations

import numpy as np

from .chem import canonicalize_smiles, molecular_descriptors
from .property_guidance import parse_property_targets, property_match_score
from .schema import BenchmarkExample, Candidate, TaskType

class RetrievalDecoder:
    """Nearest-neighbor molecular latent decoder.

    This is intentionally a baseline. It gives the JEPA planner a concrete
    molecule surface while future learned graph/SELFIES decoders are built.
    """

    def __init__(self, smiles: list[str], latents: np.ndarray, de_novo_latent_rerank_weight: float = 0.05):
        self.smiles = [canonicalize_smiles(smi) or smi for smi in smiles]
        self.latents = np.asarray(latents, dtype=np.float32)
        self.descriptors = [molecular_descriptors(smi).descriptors for smi in self.smiles]
        self.de_novo_latent_rerank_weight = float(de_novo_latent_rerank_weight)

    def decode(
        self,
        pred_latents: np.ndarray,
        source_smiles: list[str | None],
        top_k: int = 5,
        examples: list[BenchmarkExample] | None = None,
    ) -> list[list[Candidate]]:
        pred_latents = np.asarray(pred_latents, dtype=np.float32)
        sims = _cosine_similarity(pred_latents, self.latents)
        all_candidates: list[list[Candidate]] = []
        for row_idx in range(len(pred_latents)):
            example = examples[row_idx] if examples else None
            order, origin = self._candidate_order(sims[row_idx], example)
            seen: set[str] = set()
            row: list[Candidate] = []
            source = canonicalize_smiles(source_smiles[row_idx]) if source_smiles[row_idx] else None
            if source:
                seen.add(source)
            for idx in order:
                smiles = self.smiles[int(idx)]
                if smiles in seen:
                    continue
                row.append(Candidate(smiles=smiles, origin=origin, score=self._candidate_score(idx, sims[row_idx, idx], example), rank=len(row) + 1))
                seen.add(smiles)
                if len(row) >= top_k:
                    break
            all_candidates.append(row)
        return all_candidates

    def _candidate_order(self, latent_sims: np.ndarray, example: BenchmarkExample | None) -> tuple[np.ndarray, str]:
        if example and example.task_type == TaskType.DE_NOVO:
            targets = parse_property_targets(example.instruction)
            if targets:
                scores = np.asarray([property_match_score(desc, targets) for desc in self.descriptors], dtype=np.float32)
                scores = scores + self.de_novo_latent_rerank_weight * latent_sims
                return np.argsort(-scores), "property_guided_retrieval"
        return np.argsort(-latent_sims), "latent_retrieval"

    def _candidate_score(self, idx: int, latent_sim: float, example: BenchmarkExample | None) -> float:
        if example and example.task_type == TaskType.DE_NOVO:
            targets = parse_property_targets(example.instruction)
            if targets:
                return float(property_match_score(self.descriptors[int(idx)], targets) + self.de_novo_latent_rerank_weight * latent_sim)
        return float(latent_sim)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).T
    denom = np.maximum(left_norm * right_norm, 1e-8)
    return (left @ right.T) / denom
