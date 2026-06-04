"""Trajectory log schema and chemistry helpers.

The schema is intentionally file-based and simple so SketchMol sampling,
MolScribe OCR, RDKit property evaluation, and downstream diffusion training can
communicate through one durable JSONL artifact.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


PROPERTY_NAMES = ("logp", "qed", "tpsa", "molwt")


@dataclass
class TrajectoryStep:
    """One molecular optimization step."""

    trajectory_id: str
    step: int
    smiles: str
    parent_smiles: str = ""
    image_path: str = ""
    source: str = ""
    task_name: str = ""
    condition: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, float] = field(default_factory=dict)
    delta_properties: dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    validity: bool = False
    molscribe_score: float | None = None
    failure_reason: str = ""
    selected_next_action: str = ""
    edit_type: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "TrajectoryStep":
        values = dict(row)
        for key in ("condition", "properties", "delta_properties"):
            if isinstance(values.get(key), str):
                text = values[key].strip()
                values[key] = json.loads(text) if text else {}
        if values.get("molscribe_score") in ("", None):
            values["molscribe_score"] = None
        return cls(
            trajectory_id=str(values.get("trajectory_id", "")),
            step=int(values.get("step", 0)),
            smiles=str(values.get("smiles", "")),
            parent_smiles=str(values.get("parent_smiles", "")),
            image_path=str(values.get("image_path", "")),
            source=str(values.get("source", "")),
            task_name=str(values.get("task_name", "")),
            condition=dict(values.get("condition") or {}),
            properties={str(k): float(v) for k, v in dict(values.get("properties") or {}).items()},
            delta_properties={str(k): float(v) for k, v in dict(values.get("delta_properties") or {}).items()},
            reward=float(values.get("reward", 0.0) or 0.0),
            validity=bool(values.get("validity", False)),
            molscribe_score=None if values.get("molscribe_score") is None else float(values["molscribe_score"]),
            failure_reason=str(values.get("failure_reason", "")),
            selected_next_action=str(values.get("selected_next_action", "")),
            edit_type=str(values.get("edit_type", "")),
        )


def read_jsonl(path: str | Path) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                steps.append(TrajectoryStep.from_mapping(json.loads(line)))
    return steps


def write_jsonl(path: str | Path, steps: Iterable[TrajectoryStep]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for step in steps:
            handle.write(step.to_json() + "\n")


def write_csv(path: str | Path, steps: Iterable[TrajectoryStep]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field.name for field in TrajectoryStep.__dataclass_fields__.values()]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            row = asdict(step)
            for key in ("condition", "properties", "delta_properties"):
                row[key] = json.dumps(row[key], sort_keys=True)
            writer.writerow(row)


def _load_rdkit() -> tuple[Any, Any]:
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, QED

        RDLogger.DisableLog("rdApp.warning")
        return Chem, (Descriptors, QED)
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Trajectory chemistry helpers require RDKit.") from exc


def compute_properties(smiles: str) -> tuple[dict[str, float], bool, str]:
    """Return RDKit properties, validity, and failure reason."""

    Chem, descriptor_modules = _load_rdkit()
    descriptors, qed = descriptor_modules
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}, False, "rdkit_parse_failed"
    try:
        props = {
            "logp": float(descriptors.MolLogP(mol)),
            "qed": float(qed.qed(mol)),
            "tpsa": float(descriptors.TPSA(mol)),
            "molwt": float(descriptors.MolWt(mol)),
        }
    except Exception as exc:  # pragma: no cover - rare RDKit failures
        return {}, False, f"property_failed:{exc}"
    return props, True, ""


def property_delta(previous: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    return {name: float(current.get(name, 0.0) - previous.get(name, 0.0)) for name in PROPERTY_NAMES}


def reward_from_delta(delta: dict[str, float], task_name: str = "", condition: dict[str, Any] | None = None) -> float:
    """Task-aware scalar reward used for logging and lightweight benchmarks."""

    condition = condition or {}
    task = task_name.lower()
    if "qed" in task:
        return float(delta.get("qed", 0.0))
    if "tpsa" in task:
        target = condition.get("tpsa")
        if target is not None:
            before_error = abs(float(condition.get("previous_tpsa", target)) - float(target))
            after_error = abs(float(condition.get("current_tpsa", target)) - float(target))
            return float(before_error - after_error)
        return -abs(float(delta.get("tpsa", 0.0)))
    if "logp" in task or "alogp" in task:
        return float(delta.get("logp", 0.0))
    if any(name in task for name in ("akt", "ep4", "rock")):
        return float(condition.get("activity_delta", 0.0))
    return float(delta.get("qed", 0.0) + 0.1 * delta.get("logp", 0.0))


def finite_or_zero(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return 0.0
    return result if math.isfinite(result) else 0.0

