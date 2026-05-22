"""Shared schema for SketchImage-JEPA experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    DE_NOVO = "de_novo"
    EDIT = "edit"
    INPAINT = "inpaint"
    FRAGMENT_GROW = "fragment_grow"


@dataclass(frozen=True)
class BenchmarkExample:
    task_id: str
    task_type: TaskType
    instruction: str
    target_smiles: str
    source_smiles: str | None = None
    mask_hint: str | None = None
    image_path: str | None = None
    goals: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Candidate:
    smiles: str
    origin: str
    score: float = 0.0
    rank: int = 0
