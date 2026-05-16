"""Lightweight action grammar for verified molecular editing instructions."""

from __future__ import annotations

from typing import Any

from .instruction_schema import CONSTRAINTS, EDIT_RULES, PROPERTY_GOALS, normalize_spec, spec_to_json
from .instruction_templates import extract_instruction_tags, instruction_text_is_consistent


def ground_instruction_text(text: str, base_spec: str | dict[str, Any] | None = None) -> dict[str, Any]:
    """Ground natural language into the executable instruction-spec schema.

    This is intentionally conservative: it only recognizes tags from the fixed
    action grammar and never invents thresholds from free text. If a base spec
    is provided, thresholds are inherited and extra text tags are reported.
    """

    base = normalize_spec(base_spec or {})
    tags = extract_instruction_tags(text)
    goals = [tag for tag in tags if tag in PROPERTY_GOALS]
    constraints = [tag for tag in tags if tag in CONSTRAINTS]
    edits = [tag for tag in tags if tag in EDIT_RULES]
    spec = normalize_spec(
        {
            "goals": goals or base["goals"],
            "constraints": constraints or base["constraints"],
            "edits": edits or base["edits"],
            "thresholds": base["thresholds"],
        }
    )
    consistency = instruction_text_is_consistent(text, base) if base_spec is not None else {"consistent": True, "extra_tags": []}
    return {
        "instruction_text": text,
        "instruction_spec_json": spec_to_json(spec),
        "recognized_tags": tags,
        "consistent_with_base": bool(consistency["consistent"]),
        "extra_tags": list(consistency.get("extra_tags", [])),
    }


def instruction_difficulty(spec: str | dict[str, Any]) -> str:
    normalized = normalize_spec(spec)
    total_actions = len(normalized["goals"]) + len(normalized["edits"])
    constraints = len(normalized["constraints"])
    if total_actions <= 1 and constraints <= 2:
        return "easy"
    if total_actions <= 2 and constraints <= 3:
        return "medium"
    return "hard"
