#!/usr/bin/env python3
"""Canonical constraint representation shared by de novo and editing tasks."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


SCHEMA_VERSION = "unified_molecular_constraint_ir_v1"
KNOWN_PROPERTIES = (
    "MW",
    "LogP",
    "QED",
    "TPSA",
    "HBD",
    "HBA",
    "RB",
    "SA",
    "GSK3B",
    "DRD2",
    "BBBP",
    "HIA",
    "Mutagenicity",
)
PROPERTY_ALIASES = {
    "molwt": "MW",
    "molecularweight": "MW",
    "mw": "MW",
    "logp": "LogP",
    "plogp": "LogP",
    "qed": "QED",
    "tpsa": "TPSA",
    "hbd": "HBD",
    "hbonddonor": "HBD",
    "hba": "HBA",
    "hbondacceptor": "HBA",
    "rb": "RB",
    "rotatable": "RB",
    "rotatablebonds": "RB",
    "sa": "SA",
    "sascore": "SA",
    "gsk3b": "GSK3B",
    "drd2": "DRD2",
    "bbbp": "BBBP",
    "hia": "HIA",
    "mutag": "Mutagenicity",
    "mutagenicity": "Mutagenicity",
}


@dataclass(frozen=True)
class ConstraintSpec:
    property: str
    objective: str
    direction: int
    target: float | None
    threshold: float | None
    source_value: float | None
    hard: bool = True


@dataclass(frozen=True)
class MolecularConstraintIR:
    schema_version: str
    condition_id: str
    benchmark_task: str
    task_mode: str
    action_space: str
    instruction: str
    source_smiles: str
    constraints: tuple[ConstraintSpec, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def parse_json(value: object, default: object) -> object:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def parse_float(value: object) -> float | None:
    try:
        number = float(str(value or "").strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def normalize_property(value: object) -> str:
    raw = str(value or "").strip()
    key = re.sub(r"[^a-z0-9]", "", raw.lower())
    return PROPERTY_ALIASES.get(key, raw)


def parse_direction(value: object) -> int:
    text = str(value or "").strip().lower()
    if text in {"1", "+1", "+", "up", "increase", "improve", "higher", "maximize", "max"}:
        return 1
    if text in {"-1", "-", "down", "decrease", "lower", "minimize", "min"}:
        return -1
    number = parse_float(value)
    if number is not None:
        return 1 if number > 0 else -1 if number < 0 else 0
    return 0


def condition_id(row: Mapping[str, object]) -> str:
    for key in ("condition_id", "sample_id", "example_id", "variant_id", "pair_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def listed_properties(row: Mapping[str, object]) -> list[str]:
    out: list[str] = []
    for key in (
        "condition_properties",
        "instruction_task_properties",
        "external_task_properties",
        "external_supported_proxy_properties",
    ):
        for item in re.split(r"[,;+|]", str(row.get(key, "") or "")):
            prop = normalize_property(item)
            if prop and prop not in out:
                out.append(prop)
    instruction_tasks = parse_json(row.get("instruction_tasks"), [])
    if isinstance(instruction_tasks, list):
        for item in instruction_tasks:
            if isinstance(item, Mapping):
                prop = normalize_property(item.get("property") or item.get("name"))
                if prop and prop not in out:
                    out.append(prop)
    for prop in KNOWN_PROPERTIES:
        if truthy(row.get(f"{prop}_active")) and prop not in out:
            out.append(prop)
    return out


def mapping_field(row: Mapping[str, object], key: str) -> dict[str, object]:
    value = parse_json(row.get(key), {})
    if not isinstance(value, Mapping):
        return {}
    return {normalize_property(prop): item for prop, item in value.items()}


def instruction_task_map(row: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    value = parse_json(row.get("instruction_tasks"), [])
    if not isinstance(value, list):
        return {}
    out = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        prop = normalize_property(item.get("property") or item.get("name"))
        if prop:
            out[prop] = item
    return out


def build_constraint(row: Mapping[str, object], prop: str) -> ConstraintSpec:
    objectives = mapping_field(row, "external_property_objectives_json")
    directions = mapping_field(row, "external_property_directions_json")
    thresholds = mapping_field(row, "external_property_thresholds_json")
    tasks = instruction_task_map(row)
    task = tasks.get(prop, {})

    raw_objective = str(objectives.get(prop, "") or task.get("objective", "") or "").strip().lower()
    target = parse_float(row.get(f"target_{prop}"))
    source_value = parse_float(row.get(f"source_{prop}"))
    threshold = parse_float(thresholds.get(prop))
    if threshold is None:
        threshold = parse_float(task.get("threshold"))
    direction = parse_direction(
        directions.get(prop)
        or task.get("direction")
        or row.get(f"{prop}_direction")
    )

    if raw_objective in {"maintain", "preserve", "keep"}:
        objective = "maintain"
        direction = 0
    elif raw_objective in {"target", "reach", "match"} or (target is not None and direction == 0):
        objective = "target"
    elif direction:
        objective = "improve"
    elif target is not None:
        objective = "target"
    else:
        objective = "improve"

    return ConstraintSpec(
        property=prop,
        objective=objective,
        direction=direction,
        target=target,
        threshold=threshold,
        source_value=source_value,
        hard=not str(task.get("hard", "true") or "true").strip().lower() in {"0", "false", "no"},
    )


def build_constraint_ir(row: Mapping[str, object]) -> MolecularConstraintIR:
    source = str(row.get("source_smiles", "") or row.get("molecule_smiles", "") or "").strip()
    explicit_mode = str(row.get("task_mode", "") or "").strip().lower()
    task_mode = "edit" if explicit_mode == "edit" or source else "de_novo"
    action_space = "graph_edit_dsl" if task_mode == "edit" else "smiles"
    instruction = str(row.get("instruction", "") or row.get("prompt", "") or "").strip()
    benchmark = str(
        row.get("benchmark_task", "")
        or row.get("external_suite", "")
        or row.get("task_type", "")
        or "unknown"
    ).strip()
    constraints = tuple(build_constraint(row, prop) for prop in listed_properties(row))
    return MolecularConstraintIR(
        schema_version=SCHEMA_VERSION,
        condition_id=condition_id(row),
        benchmark_task=benchmark,
        task_mode=task_mode,
        action_space=action_space,
        instruction=instruction,
        source_smiles=source,
        constraints=constraints,
    )


def constraint_property_names(ir: MolecularConstraintIR) -> Sequence[str]:
    return tuple(item.property for item in ir.constraints)
