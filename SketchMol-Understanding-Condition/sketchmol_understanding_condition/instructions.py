"""Instruction templates for molecular image editing pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class EditInstruction:
    """Structured fields used to render a natural-language edit instruction."""

    task: str
    scaffold_policy: str = "preserve"
    source_group: Optional[str] = None
    target_group: Optional[str] = None
    property_name: Optional[str] = None
    property_direction: Optional[str] = None
    protein_target: Optional[str] = None


_TASK_TEMPLATES = {
    "functional_group_replacement": (
        "Keep the core scaffold unchanged and replace {source_group} with "
        "{target_group}."
    ),
    "property_optimization": (
        "Keep the core scaffold recognizable and modify the side chain to "
        "{property_direction} {property_name}."
    ),
    "protein_aware_optimization": (
        "Preserve the molecular scaffold, improve {protein_target} binding, "
        "and {property_direction} {property_name} if possible."
    ),
    "scaffold_preserving_edit": (
        "Preserve the core scaffold and make a local chemical edit that "
        "keeps the molecule valid."
    ),
}


def render_instruction(
    instruction: EditInstruction | Mapping[str, object],
    *,
    fallback: str = "Preserve the core scaffold and improve the requested molecular property.",
) -> str:
    """Render a structured instruction into a deterministic text prompt."""

    if not isinstance(instruction, EditInstruction):
        instruction = EditInstruction(**dict(instruction))

    template = _TASK_TEMPLATES.get(instruction.task)
    if template is None:
        return fallback

    values = {
        "source_group": instruction.source_group or "the source functional group",
        "target_group": instruction.target_group or "the requested functional group",
        "property_name": instruction.property_name or "the target property",
        "property_direction": instruction.property_direction or "improve",
        "protein_target": instruction.protein_target or "the target protein",
    }
    return template.format(**values)


def property_direction_from_delta(delta: float, eps: float = 1e-8) -> str:
    """Convert a target-source property delta into an instruction direction."""

    if delta > eps:
        return "increase"
    if delta < -eps:
        return "decrease"
    return "maintain"
