"""Shared helpers for multi-property dataset construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping


REPO_DIR = Path(__file__).resolve().parents[2]
UNDERSTANDING_DIR = REPO_DIR / "SketchMol-Understanding-Condition"
if str(UNDERSTANDING_DIR) not in sys.path:
    sys.path.insert(0, str(UNDERSTANDING_DIR))


PROPERTY_COLUMNS = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB")

PROPERTY_ALIASES = {
    "MW": "MW",
    "MolWt": "MW",
    "LogP": "LogP",
    "QED": "QED",
    "TPSA": "TPSA",
    "HBD": "HBD",
    "HBA": "HBA",
    "RB": "RB",
    "rotatable": "RB",
}

DELTA_THRESHOLDS = {
    "MW": 35.0,
    "LogP": 0.5,
    "QED": 0.05,
    "TPSA": 15.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
}

SKETCHMOL_STRICT_TOLERANCE = {
    "MW": 35.0,
    "LogP": 1.0,
    "QED": 0.10,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
}

SKETCHMOL_REFERENCE_MULTI_PROPERTY = {
    2: 0.804,
    3: 0.768,
    4: 0.736,
    5: 0.716,
    6: 0.678,
    7: 0.685,
}

SKETCHMOL_SETTING_COLUMNS = {
    "MW": ("MolWt_setting", "MolWt_None"),
    "LogP": ("logp_setting", "logp_None"),
    "QED": ("QED_setting", "QED_None"),
    "TPSA": ("TPSA_setting", "TPSA_None"),
    "HBD": ("HBD_setting", "HBD_None"),
    "HBA": ("HBA_setting", "HBA_None"),
    "RB": ("rotatable_setting", "rotatable_None"),
}

DISPLAY_NAMES = {
    "MW": "molecular weight",
    "LogP": "LogP",
    "QED": "QED",
    "TPSA": "TPSA",
    "HBD": "hydrogen-bond donor count",
    "HBA": "hydrogen-bond acceptor count",
    "RB": "rotatable bond count",
}


def normalize_properties(raw: Mapping[str, object]) -> dict[str, float]:
    """Normalize common property aliases into SketchMol's 7-property schema."""

    out: dict[str, float] = {}
    for key, value in raw.items():
        prop = PROPERTY_ALIASES.get(key)
        if prop is None:
            continue
        try:
            out[prop] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def active_property_deltas(
    source: Mapping[str, float],
    target: Mapping[str, float],
    *,
    threshold_scale: float = 1.0,
) -> dict[str, float]:
    """Return property deltas whose absolute change passes configured thresholds."""

    active = {}
    for prop in PROPERTY_COLUMNS:
        if prop not in source or prop not in target:
            continue
        delta = float(target[prop]) - float(source[prop])
        threshold = DELTA_THRESHOLDS[prop] * float(threshold_scale)
        if abs(delta) >= threshold:
            active[prop] = delta
    return active


def direction_from_delta(delta: float) -> str:
    return "increase" if float(delta) >= 0 else "decrease"


def format_float(value: float, digits: int = 4) -> str:
    text = f"{float(value):.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sketchmol_condition_columns(target_props: Mapping[str, float], selected_props: list[str]) -> dict[str, object]:
    """Build SketchMolBenchmark-style property setting/None columns."""

    selected = set(selected_props)
    out: dict[str, object] = {}
    for prop in PROPERTY_COLUMNS:
        value_col, none_col = SKETCHMOL_SETTING_COLUMNS[prop]
        if prop in selected:
            out[value_col] = target_props.get(prop, "")
            out[none_col] = "False"
        else:
            out[value_col] = ""
            out[none_col] = "True"
    return out


def render_instruction(
    *,
    selected_props: list[str],
    source_props: Mapping[str, float],
    target_props: Mapping[str, float],
    deltas: Mapping[str, float],
) -> str:
    """Render a compact natural-language multi-property edit instruction."""

    clauses = []
    for prop in selected_props:
        delta = float(deltas[prop])
        direction = direction_from_delta(delta)
        target = target_props[prop]
        name = DISPLAY_NAMES[prop]
        if prop in {"HBD", "HBA", "RB"}:
            clauses.append(f"{direction} {name} toward {format_float(target, 0)}")
        elif prop == "QED":
            clauses.append(f"{direction} {name} toward {format_float(target, 3)}")
        else:
            clauses.append(f"{direction} {name} toward {format_float(target, 2)}")
    if len(clauses) == 1:
        objective_text = clauses[0]
    elif len(clauses) == 2:
        objective_text = f"{clauses[0]} and {clauses[1]}"
    else:
        objective_text = f"{', '.join(clauses[:-1])}, and {clauses[-1]}"
    return f"Preserve the core scaffold and edit the molecule to {objective_text}."
