"""Dependency-free P31.1 reward and leave-one-out advantage helpers."""

from __future__ import annotations

from typing import Mapping, Sequence


def scalar_reward(channels: Mapping[str, float], details: Mapping[str, object], mode: str) -> float:
    """Bounded reward whose ordering is dominated by the paper's strict metric."""
    del channels
    if not bool(details.get("valid")):
        return -1.0
    canonical = float(bool(details.get("canonical")))
    mean_satisfaction = float(details.get("mean_satisfaction", 0.0))
    bottleneck = float(details.get("bottleneck", 0.0))
    fraction = float(details.get("property_fraction", 0.0))
    base = 0.25 * canonical + 0.75 * mean_satisfaction + bottleneck + 0.50 * fraction
    if mode == "de_novo":
        if bool(details.get("strict")):
            return 3.0 + 0.50 * mean_satisfaction + 0.50 * bottleneck
        return base
    similarity = float(details.get("source_similarity") or 0.0)
    if bool(details.get("strict")):
        return (
            4.0 + 0.50 * mean_satisfaction + 0.50 * bottleneck
            + 0.25 * min(max(similarity, 0.0) / 0.65, 1.0)
            - 0.25 * float(bool(details.get("copy")))
        )
    if bool(details.get("property_strict")):
        return 2.5 + 0.50 * min(max(similarity, 0.0) / 0.65, 1.0)
    return base


def rloo_advantages(returns: Sequence[float]) -> list[float]:
    if len(returns) < 2:
        raise ValueError("RLOO requires at least two rollouts")
    total = sum(float(value) for value in returns)
    denominator = len(returns) - 1
    return [float(value) - (total - float(value)) / denominator for value in returns]
