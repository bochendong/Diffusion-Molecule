#!/usr/bin/env python3
"""Dependency-free terminal-return group math for P32.2."""

from __future__ import annotations

from typing import Sequence


def centered_advantages(returns: Sequence[float], clip: float = 3.0) -> list[float]:
    if not returns:
        return []
    center = sum(float(value) for value in returns) / len(returns)
    variance = sum((float(value) - center) ** 2 for value in returns) / len(returns)
    scale = variance**0.5
    if scale < 1e-8:
        return [0.0 for _value in returns]
    return [max(-clip, min(clip, (float(value) - center) / scale)) for value in returns]
