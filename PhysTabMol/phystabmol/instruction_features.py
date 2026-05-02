"""Feature builders for instruction-guided edit planning."""

from __future__ import annotations

from typing import Any

import numpy as np

from .features import table_row_from_smiles
from .instruction_schema import CONSTRAINTS, EDIT_NAMES, GOAL_NAMES, DEFAULT_THRESHOLDS, PROPERTY_GOALS, normalize_spec, threshold
from .schema import TABLE_COLUMNS, TARGET_COLUMNS


THRESHOLD_FEATURES = (
    "delta_logp_min",
    "delta_qed_min",
    "delta_tpsa_min",
    "delta_mw_min",
    "delta_mw_abs_max",
    "delta_hba_min",
    "delta_hbd_min",
    "delta_rb_min",
    "delta_sa_min",
    "delta_halogen_min",
    "delta_heteroatom_min",
    "delta_fg_min",
    "similarity_min",
)

INSTRUCTION_SPEC_FEATURE_START = len(TABLE_COLUMNS) + len(TARGET_COLUMNS)


def instruction_feature_names() -> list[str]:
    return (
        [f"source_{col}" for col in TABLE_COLUMNS]
        + [f"target_hint_{col}" for col in TARGET_COLUMNS]
        + [f"goal_{name}" for name in GOAL_NAMES]
        + [f"constraint_{name}" for name in CONSTRAINTS]
        + [f"edit_{name}" for name in EDIT_NAMES]
        + [f"threshold_{name}" for name in THRESHOLD_FEATURES]
    )


INSTRUCTION_SPEC_FEATURE_END = len(instruction_feature_names())


def condition_from_source_and_spec(source_smiles: str, spec: str | dict[str, Any]) -> np.ndarray:
    source_row = table_row_from_smiles(source_smiles)
    normalized = normalize_spec(spec)
    target_hints = target_hints_from_spec(source_row, normalized)
    features = []
    features.extend(float(source_row[col]) for col in TABLE_COLUMNS)
    features.extend(float(target_hints[col]) for col in TARGET_COLUMNS)
    features.extend(1.0 if name in normalized["goals"] else 0.0 for name in GOAL_NAMES)
    features.extend(1.0 if name in normalized["constraints"] else 0.0 for name in CONSTRAINTS)
    features.extend(1.0 if name in normalized["edits"] else 0.0 for name in EDIT_NAMES)
    for name in THRESHOLD_FEATURES:
        default = DEFAULT_THRESHOLDS.get(name, 1.0)
        scale = max(abs(default), 1.0)
        features.append(float(threshold(normalized, name, default)) / scale)
    return np.asarray(features, dtype=np.float32)


def target_hints_from_spec(source_row: dict[str, float], spec: str | dict[str, Any]) -> dict[str, float]:
    normalized = normalize_spec(spec)
    hints = {col: float(source_row[col]) for col in TARGET_COLUMNS}
    for goal in normalized["goals"]:
        rule = PROPERTY_GOALS.get(goal)
        if rule is None:
            continue
        col = rule["column"]
        if col not in hints:
            continue
        hints[col] += float(rule["direction"]) * threshold(normalized, rule["threshold"], rule["default"])
    if "keep_mw_similar" in normalized["constraints"]:
        hints["MW"] = float(source_row["MW"])
    return hints


def target_table_from_smiles(smiles: str) -> np.ndarray:
    row = table_row_from_smiles(smiles)
    return np.asarray([float(row[col]) for col in TABLE_COLUMNS], dtype=np.float32)
