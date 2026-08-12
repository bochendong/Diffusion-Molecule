#!/usr/bin/env python3
"""Stable-anchor ranking primitives for bounded common-LLM residual policies."""

from __future__ import annotations

from typing import Sequence


def anchored_order(
    reference_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    anchor_top_k: int,
    max_residual_rank_shift: float,
) -> list[int]:
    """Preserve the reference top-k set and allow bounded reranking inside it."""

    if len(reference_scores) != len(candidate_scores):
        raise ValueError("Reference/candidate score lengths differ")
    if not reference_scores:
        return []
    if not 1 <= int(anchor_top_k) <= len(reference_scores):
        raise ValueError("anchor_top_k is outside the candidate pool")
    reference_order = sorted(
        range(len(reference_scores)),
        key=lambda index: (float(reference_scores[index]), -index),
        reverse=True,
    )
    protected = reference_order[: int(anchor_top_k)]
    residuals = {
        index: float(candidate_scores[index]) - float(reference_scores[index])
        for index in protected
    }
    residual_order = sorted(protected, key=lambda index: (residuals[index], -index))
    percentiles = {index: 0.5 for index in protected}
    if len(protected) > 1:
        for rank, index in enumerate(residual_order):
            percentiles[index] = rank / float(len(protected) - 1)
    reference_ranks = {index: rank for rank, index in enumerate(reference_order)}
    protected = sorted(
        protected,
        key=lambda index: (
            -float(reference_ranks[index])
            + float(max_residual_rank_shift) * (percentiles[index] - 0.5),
            float(candidate_scores[index]),
            float(reference_scores[index]),
            -index,
        ),
        reverse=True,
    )
    protected_set = set(protected)
    return [*protected, *(index for index in reference_order if index not in protected_set)]


def top_k_set(scores: Sequence[float], k: int) -> set[int]:
    return set(
        sorted(
            range(len(scores)),
            key=lambda index: (float(scores[index]), -index),
            reverse=True,
        )[: int(k)]
    )
