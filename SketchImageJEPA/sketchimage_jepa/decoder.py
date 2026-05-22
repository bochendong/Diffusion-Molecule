"""Latent-to-molecule decoding baselines."""

from __future__ import annotations

import numpy as np

from .chem import canonicalize_smiles
from .schema import Candidate


class RetrievalDecoder:
    """Nearest-neighbor molecular latent decoder.

    This is intentionally a baseline. It gives the JEPA planner a concrete
    molecule surface while future learned graph/SELFIES decoders are built.
    """

    def __init__(self, smiles: list[str], latents: np.ndarray):
        self.smiles = [canonicalize_smiles(smi) or smi for smi in smiles]
        self.latents = np.asarray(latents, dtype=np.float32)

    def decode(self, pred_latents: np.ndarray, source_smiles: list[str | None], top_k: int = 5) -> list[list[Candidate]]:
        pred_latents = np.asarray(pred_latents, dtype=np.float32)
        sims = _cosine_similarity(pred_latents, self.latents)
        all_candidates: list[list[Candidate]] = []
        for row_idx in range(len(pred_latents)):
            order = np.argsort(-sims[row_idx])
            seen: set[str] = set()
            row: list[Candidate] = []
            source = canonicalize_smiles(source_smiles[row_idx]) if source_smiles[row_idx] else None
            if source:
                row.append(Candidate(smiles=source, origin="source_anchor", score=float(sims[row_idx, order[0]]), rank=1))
                seen.add(source)
            for idx in order:
                smiles = self.smiles[int(idx)]
                if smiles in seen:
                    continue
                row.append(Candidate(smiles=smiles, origin="latent_retrieval", score=float(sims[row_idx, idx]), rank=len(row) + 1))
                seen.add(smiles)
                if len(row) >= top_k:
                    break
            all_candidates.append(row)
        return all_candidates


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).T
    denom = np.maximum(left_norm * right_norm, 1e-8)
    return (left @ right.T) / denom
