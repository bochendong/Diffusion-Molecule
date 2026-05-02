"""Schema helpers for verified instruction-guided molecular editing.

The instruction benchmark deliberately keeps the chemistry judge deterministic:
LLMs may paraphrase instructions, but the executable spec below is what RDKit
and rule-based checks verify.
"""

from __future__ import annotations

import json
from typing import Any


SPEC_VERSION = 1

PROPERTY_GOALS: dict[str, dict[str, Any]] = {
    "increase_logp": {"column": "LogP", "direction": 1, "threshold": "delta_logp_min", "default": 0.5},
    "decrease_logp": {"column": "LogP", "direction": -1, "threshold": "delta_logp_min", "default": 0.5},
    "improve_qed": {"column": "QED", "direction": 1, "threshold": "delta_qed_min", "default": 0.05},
    "reduce_tpsa": {"column": "TPSA", "direction": -1, "threshold": "delta_tpsa_min", "default": 15.0},
    "increase_tpsa": {"column": "TPSA", "direction": 1, "threshold": "delta_tpsa_min", "default": 15.0},
    "increase_mw": {"column": "MW", "direction": 1, "threshold": "delta_mw_min", "default": 25.0},
    "decrease_mw": {"column": "MW", "direction": -1, "threshold": "delta_mw_min", "default": 25.0},
    "increase_hba": {"column": "HBA", "direction": 1, "threshold": "delta_hba_min", "default": 1.0},
    "increase_hbd": {"column": "HBD", "direction": 1, "threshold": "delta_hbd_min", "default": 1.0},
    "decrease_rb": {"column": "RB", "direction": -1, "threshold": "delta_rb_min", "default": 1.0},
    "lower_sa": {"column": "SA", "direction": -1, "threshold": "delta_sa_min", "default": 0.35},
}

CONSTRAINTS: tuple[str, ...] = (
    "preserve_scaffold",
    "keep_mw_similar",
    "keep_similarity",
    "keep_druglike",
)

EDIT_RULES: dict[str, dict[str, Any]] = {
    "add_halogen": {"column": "fg_halogen", "direction": 1, "threshold": "delta_halogen_min", "default": 1.0},
    "remove_halogen": {"column": "fg_halogen", "direction": -1, "threshold": "delta_halogen_min", "default": 1.0},
    "add_heteroatom": {"columns": ("N", "O", "S"), "direction": 1, "threshold": "delta_heteroatom_min", "default": 1.0},
    "reduce_heteroatom": {"columns": ("N", "O", "S"), "direction": -1, "threshold": "delta_heteroatom_min", "default": 1.0},
    "increase_hba": {"column": "HBA", "direction": 1, "threshold": "delta_hba_min", "default": 1.0},
    "increase_hbd": {"column": "HBD", "direction": 1, "threshold": "delta_hbd_min", "default": 1.0},
    "add_ester": {"column": "fg_ester", "direction": 1, "threshold": "delta_fg_min", "default": 1.0},
    "remove_ester": {"column": "fg_ester", "direction": -1, "threshold": "delta_fg_min", "default": 1.0},
    "add_amide": {"column": "fg_amide", "direction": 1, "threshold": "delta_fg_min", "default": 1.0},
    "remove_amide": {"column": "fg_amide", "direction": -1, "threshold": "delta_fg_min", "default": 1.0},
    "add_amine": {"column": "fg_amine", "direction": 1, "threshold": "delta_fg_min", "default": 1.0},
    "remove_amine": {"column": "fg_amine", "direction": -1, "threshold": "delta_fg_min", "default": 1.0},
    "add_alcohol": {"column": "fg_alcohol", "direction": 1, "threshold": "delta_fg_min", "default": 1.0},
    "remove_alcohol": {"column": "fg_alcohol", "direction": -1, "threshold": "delta_fg_min", "default": 1.0},
}

DEFAULT_THRESHOLDS: dict[str, float] = {
    "delta_logp_min": 0.5,
    "delta_qed_min": 0.05,
    "delta_tpsa_min": 15.0,
    "delta_mw_min": 25.0,
    "delta_mw_abs_max": 80.0,
    "delta_hba_min": 1.0,
    "delta_hbd_min": 1.0,
    "delta_rb_min": 1.0,
    "delta_sa_min": 0.35,
    "delta_halogen_min": 1.0,
    "delta_heteroatom_min": 1.0,
    "delta_fg_min": 1.0,
    "similarity_min": 0.6,
}

GOAL_NAMES = tuple(PROPERTY_GOALS)
EDIT_NAMES = tuple(EDIT_RULES)
TAG_NAMES = GOAL_NAMES + CONSTRAINTS + EDIT_NAMES


def normalize_spec(spec: str | dict[str, Any]) -> dict[str, Any]:
    """Return a compact, stable instruction spec dictionary."""

    if isinstance(spec, str):
        spec = json.loads(spec)
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update({str(k): float(v) for k, v in dict(spec.get("thresholds", {})).items()})
    return {
        "version": int(spec.get("version", SPEC_VERSION)),
        "goals": _dedupe_known(spec.get("goals", []), PROPERTY_GOALS),
        "constraints": _dedupe_known(spec.get("constraints", []), CONSTRAINTS),
        "edits": _dedupe_known(spec.get("edits", []), EDIT_RULES),
        "thresholds": thresholds,
    }


def spec_to_json(spec: str | dict[str, Any]) -> str:
    return json.dumps(normalize_spec(spec), sort_keys=True, separators=(",", ":"))


def threshold(spec: dict[str, Any], key: str, default: float | None = None) -> float:
    normalized = normalize_spec(spec)
    if default is None:
        default = DEFAULT_THRESHOLDS.get(key, 0.0)
    return float(normalized["thresholds"].get(key, default))


def spec_tags(spec: str | dict[str, Any]) -> list[str]:
    normalized = normalize_spec(spec)
    return list(normalized["goals"]) + list(normalized["constraints"]) + list(normalized["edits"])


def _dedupe_known(values: Any, allowed: Any) -> list[str]:
    allowed_set = set(allowed)
    out = []
    for value in values or []:
        item = str(value)
        if item in allowed_set and item not in out:
            out.append(item)
    return out
