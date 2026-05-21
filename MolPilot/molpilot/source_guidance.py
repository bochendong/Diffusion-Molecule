"""Source-locked candidate generation inspired by UniVideo-style editing.

UniVideo keeps conditioning latents in the denoising stream and updates only
the unknown/edit region. MolPilot does not yet have graph-region latents, so
this module implements a first source-locked approximation for edit/inpaint:
decode raw diffusion samples, decode source-anchored latent interpolations, and
add local source-neighborhood candidates for verifier-aware ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .chem import canonicalize_smiles
from .graph_editor import generate_graph_edit_candidates, generate_scaffold_library_candidates
from .schema import GenerationRequest, ObjectiveSpec, TaskType


@dataclass
class Candidate:
    smiles: str
    origin: str


def decode_source_guided_candidates(
    codec,
    request: GenerationRequest,
    latents: np.ndarray,
    objective: ObjectiveSpec | None = None,
    top_k: int = 4,
    source_edit_strengths: Iterable[float] = (0.25, 0.50),
    source_neighborhood_k: int = 32,
    graph_edit_limit: int = 96,
    scaffold_library_k: int = 32,
    enable_source_guidance: bool = True,
    include_diffusion_candidates: bool = True,
    enable_latent_source_guidance: bool = True,
    enable_graph_editor: bool = True,
) -> list[Candidate]:
    """Decode candidates with optional source-locked variants.

    ``source_edit_strength`` is the fraction of the generated delta kept:
    0.0 means the source latent is fully locked; 1.0 means the raw generated
    latent is used. Values between them approximate masked/source-preserving
    denoising before MolPilot has explicit graph-region tokens.
    """

    latents = np.asarray(latents, dtype=np.float32)
    candidates: list[Candidate] = []
    source_smiles = canonicalize_smiles(request.source_smiles) if request.source_smiles else None
    source_enabled = (
        enable_source_guidance
        and bool(source_smiles)
        and request.task_type in {TaskType.EDIT, TaskType.INPAINT}
    )

    if include_diffusion_candidates:
        for latent in latents:
            _extend_decoded(candidates, codec, latent, top_k=top_k, origin="diffusion", source_smiles=source_smiles)

    if source_enabled:
        source_latent = _encode_source(codec, source_smiles)
        if source_latent is not None and enable_latent_source_guidance:
            for strength in source_edit_strengths:
                strength = _clamp_strength(strength)
                for latent in latents:
                    guided = source_latent + strength * (latent - source_latent)
                    _extend_decoded(
                        candidates,
                        codec,
                        guided,
                        top_k=top_k,
                        origin=f"source_guided_{strength:.2f}",
                        source_smiles=source_smiles,
                    )
            if source_neighborhood_k > 0:
                _extend_decoded(
                    candidates,
                    codec,
                    source_latent,
                    top_k=source_neighborhood_k,
                    origin="source_neighborhood",
                    source_smiles=source_smiles,
                )
        if enable_graph_editor and objective is not None:
            for candidate in generate_graph_edit_candidates(source_smiles, objective, limit=graph_edit_limit):
                candidates.append(Candidate(candidate.smiles, candidate.origin))
            train_smiles = getattr(codec, "train_smiles", []) or []
            for candidate in generate_scaffold_library_candidates(
                source_smiles,
                objective,
                train_smiles,
                limit=scaffold_library_k,
            ):
                candidates.append(Candidate(candidate.smiles, candidate.origin))

    return _dedupe_candidates(candidates)


def parse_strengths(value: str | Iterable[float] | None) -> list[float]:
    if value is None:
        return [0.25, 0.50]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [_clamp_strength(float(part)) for part in parts] or [0.25, 0.50]
    return [_clamp_strength(float(part)) for part in value]


def _encode_source(codec, source_smiles: str) -> np.ndarray | None:
    if hasattr(codec, "encode_many"):
        return np.asarray(codec.encode_many([source_smiles])[0], dtype=np.float32)
    if hasattr(codec, "encode"):
        return np.asarray(codec.encode(source_smiles), dtype=np.float32)
    return None


def _extend_decoded(
    out: list[Candidate],
    codec,
    latent: np.ndarray,
    top_k: int,
    origin: str,
    source_smiles: str | None,
) -> None:
    for smiles in codec.decode(latent, top_k=max(1, top_k)):
        canon = canonicalize_smiles(smiles)
        if not canon or (source_smiles and canon == source_smiles):
            continue
        out.append(Candidate(canon, origin))


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    seen: dict[str, int] = {}
    for candidate in candidates:
        if candidate.smiles in seen:
            existing = out[seen[candidate.smiles]]
            origins = existing.origin.split("+")
            if candidate.origin not in origins:
                existing.origin = existing.origin + "+" + candidate.origin
            continue
        seen[candidate.smiles] = len(out)
        out.append(candidate)
    return out


def _clamp_strength(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
