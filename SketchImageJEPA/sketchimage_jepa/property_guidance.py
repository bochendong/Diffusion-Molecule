"""Property parsing and scoring for de novo molecular generation."""

from __future__ import annotations

import re
from typing import Mapping

PROPERTY_KEYS = ("MW", "LogP", "QED", "TPSA")
PROPERTY_TOLERANCES = {
    "MW": 60.0,
    "LogP": 0.6,
    "QED": 0.12,
    "TPSA": 20.0,
}


def parse_property_targets(text: str) -> dict[str, float]:
    targets: dict[str, float] = {}
    for key in PROPERTY_KEYS:
        match = re.search(rf"\b{re.escape(key)}\b\s*(?:around|=|:)?\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if match:
            targets[key] = float(match.group(1))
    return targets


def property_mae(candidate: Mapping[str, float], target: Mapping[str, float]) -> float:
    errors = normalized_property_errors(candidate, target)
    return float(sum(errors.values()) / len(errors)) if errors else 0.0


def property_success(candidate: Mapping[str, float], target: Mapping[str, float]) -> bool:
    errors = normalized_property_errors(candidate, target)
    return bool(errors) and all(value <= 1.0 for value in errors.values())


def property_match_score(candidate: Mapping[str, float], target: Mapping[str, float]) -> float:
    return max(0.0, 1.0 - property_mae(candidate, target))


def normalized_property_errors(candidate: Mapping[str, float], target: Mapping[str, float]) -> dict[str, float]:
    errors = {}
    for key in PROPERTY_KEYS:
        if key not in target:
            continue
        tolerance = PROPERTY_TOLERANCES[key]
        errors[key] = abs(float(candidate.get(key, 0.0)) - float(target[key])) / tolerance
    return errors
