"""Source-constrained reward used by the P32.4 editing-only online RLOO run."""

from __future__ import annotations

from typing import Mapping


def source_constrained_reward(
    channels: Mapping[str, float], details: Mapping[str, object], mode: str
) -> float:
    """Reward source preservation densely while keeping strict success dominant."""
    if mode != "edit":
        raise ValueError("P32.4 is an editing-only RL protocol")
    if not bool(details.get("valid")):
        return -2.0

    similarity = min(max(float(details.get("source_similarity") or 0.0), 0.0), 1.0)
    similarity_progress = min(similarity / 0.65, 1.0)
    property_progress = (
        0.25 * float(details.get("mean_satisfaction", 0.0))
        + 0.25 * float(details.get("bottleneck", 0.0))
        + 0.25 * float(details.get("property_fraction", 0.0))
    )
    reward = (
        0.15 * float(bool(details.get("canonical")))
        + 1.50 * similarity_progress
        + 0.40 * float(channels.get("source_aligned", 0.0))
        + property_progress
        - 1.00 * float(bool(details.get("copy")))
    )
    if bool(details.get("property_strict")):
        reward += 0.75
    if bool(details.get("relaxed")):
        reward += 0.50
    if bool(details.get("strict")):
        reward += 4.00
    return reward
