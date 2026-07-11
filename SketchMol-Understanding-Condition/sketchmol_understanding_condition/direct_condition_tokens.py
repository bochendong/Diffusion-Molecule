"""Direct-SMILES condition token builders shared with the unified generator."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from sketchmol_understanding_condition.unified_condition_dataset import PROPERTY_COLUMNS

PROPERTY_NORMALIZERS = {
    "MW": 500.0,
    "LogP": 6.0,
    "QED": 1.0,
    "TPSA": 160.0,
    "HBD": 8.0,
    "HBA": 12.0,
    "RB": 12.0,
    "SA": 8.0,
}
SKETCHMOL_STRICT_TOLERANCE = {
    "MW": 35.0,
    "LogP": 1.0,
    "QED": 0.10,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
    "SA": 1.0,
}


def parse_float(value: object) -> float:
    try:
        text = str(value).strip()
        if not text:
            return math.nan
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def truthy(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return None


def selected_properties(row: Mapping[str, str]) -> list[str]:
    selected = [item.strip() for item in str(row.get("condition_properties", "") or "").split(",") if item.strip()]
    selected = [prop for prop in selected if prop in PROPERTY_COLUMNS]
    if selected:
        return selected
    return [prop for prop in PROPERTY_COLUMNS if truthy(row.get(f"{prop}_active"))]


def row_property_count(row: Mapping[str, str]) -> int:
    explicit = parse_float(row.get("property_count"))
    if not math.isnan(explicit) and explicit > 0:
        return max(1, int(round(explicit)))
    selected = selected_properties(row)
    if selected:
        return len(selected)
    return 1


def parse_direction_value(value: object) -> int:
    text = str(value or "").strip().lower()
    if text in {"increase", "up", "+", "higher"}:
        return 1
    if text in {"decrease", "down", "-", "lower"}:
        return -1
    return 0


def expand_condition_token(values: Sequence[float], condition_dim: int) -> np.ndarray:
    source = np.asarray(list(values), dtype=np.float32)
    if source.size == 0:
        return np.zeros(max(1, int(condition_dim)), dtype=np.float32)
    repeats = int(math.ceil(max(1, int(condition_dim)) / max(source.size, 1)))
    tiled = np.tile(source, repeats)[: max(1, int(condition_dim))]
    return tiled.astype(np.float32)


def fallback_condition_features(row: Mapping[str, str], condition_dim: int) -> np.ndarray:
    values = []
    active_props = {part.strip() for part in str(row.get("condition_properties", "") or "").split(",") if part.strip()}
    for prop in PROPERTY_COLUMNS:
        value = parse_float(row.get(f"target_{prop}"))
        normalizer = PROPERTY_NORMALIZERS.get(prop, 1.0)
        values.append(0.0 if math.isnan(value) else float(value) / normalizer)
    for prop in PROPERTY_COLUMNS:
        active = truthy(row.get(f"{prop}_active"))
        values.append(1.0 if (active if active is not None else prop in active_props) else 0.0)
    for prop in PROPERTY_COLUMNS:
        direction = str(row.get(f"{prop}_direction", "") or "").strip().lower()
        values.append(1.0 if direction in {"increase", "up", "+", "higher"} else (-1.0 if direction else 0.0))
    values.append(float(len(active_props)) / max(len(PROPERTY_COLUMNS), 1))
    vec = np.zeros(max(1, int(condition_dim)), dtype=np.float32)
    source = np.asarray(values, dtype=np.float32)
    vec[: min(vec.shape[0], source.shape[0])] = source[: vec.shape[0]]
    return vec[None, :]


def property_program_tokens(row: Mapping[str, str], condition_dim: int) -> np.ndarray:
    selected = selected_properties(row)
    selected_set = set(selected)
    count = row_property_count(row)
    count_norm = float(count) / max(len(PROPERTY_COLUMNS), 1)
    directions = [parse_direction_value(row.get(f"{prop}_direction")) for prop in PROPERTY_COLUMNS]
    positive_direction_fraction = sum(1 for value in directions if value > 0) / max(len(PROPERTY_COLUMNS), 1)
    negative_direction_fraction = sum(1 for value in directions if value < 0) / max(len(PROPERTY_COLUMNS), 1)
    normalized_targets = []
    for prop in PROPERTY_COLUMNS:
        target = parse_float(row.get(f"target_{prop}"))
        normalizer = PROPERTY_NORMALIZERS.get(prop, 1.0)
        if not math.isnan(target):
            normalized_targets.append(float(target) / max(normalizer, 1e-8))
    tokens = [
        expand_condition_token(
            [
                0.25,
                count_norm,
                float(len(selected_set)) / max(len(PROPERTY_COLUMNS), 1),
                sum(normalized_targets) / len(normalized_targets) if normalized_targets else 0.0,
                max(normalized_targets) if normalized_targets else 0.0,
                min(normalized_targets) if normalized_targets else 0.0,
                positive_direction_fraction,
                negative_direction_fraction,
            ],
            condition_dim,
        )
    ]
    for idx, prop in enumerate(PROPERTY_COLUMNS):
        target = parse_float(row.get(f"target_{prop}"))
        normalizer = PROPERTY_NORMALIZERS.get(prop, 1.0)
        tolerance = float(SKETCHMOL_STRICT_TOLERANCE.get(prop, normalizer))
        tokens.append(
            expand_condition_token(
                [
                    1.0,
                    float(idx + 1) / max(len(PROPERTY_COLUMNS), 1),
                    0.0 if math.isnan(target) else float(target) / max(normalizer, 1e-8),
                    1.0 if prop in selected_set else 0.0,
                    float(parse_direction_value(row.get(f"{prop}_direction"))),
                    tolerance / max(normalizer, 1e-8),
                    count_norm,
                    0.0 if math.isnan(target) else 1.0,
                ],
                condition_dim,
            )
        )
    return np.stack(tokens, axis=0).astype(np.float32)
