#!/usr/bin/env python3
"""Shared prompt and payload contract for executable GraphEditDSL plans."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from molecular_constraint_ir import build_constraint_ir


PLAN_SYSTEM_PROMPT = (
    "You are a unified molecular planning agent. Read the constraint IR and rank an already executable "
    "GraphEditDSL plan. Return exactly one JSON object with action_type=graph_edit_plan and a value.steps "
    "list containing one or two GraphEditDSL actions. Prefer plans that satisfy every requested property "
    "while preserving the source molecule. Do not add prose or markdown."
)

MODEL_ACTION_FIELDS = (
    "op",
    "site",
    "bond",
    "atom",
    "fragment",
    "bond_order",
    "prop",
    "direction",
    "reason",
)


def model_action(action: Mapping[str, object]) -> dict[str, object]:
    """Remove planner-only scores and keep the executable semantic action."""
    output = {key: action.get(key) for key in MODEL_ACTION_FIELDS}
    output["op"] = str(output.get("op", "") or "")
    for key in ("atom", "fragment", "bond_order", "prop", "direction", "reason"):
        output[key] = str(output.get(key, "") or "")
    return output


def plan_payload(actions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    steps = [model_action(action) for action in actions]
    if not 1 <= len(steps) <= 2:
        raise ValueError(f"GraphEditDSL plan must contain one or two actions, received {len(steps)}")
    if any(not str(action.get("op", "") or "").strip() for action in steps):
        raise ValueError("GraphEditDSL plan contains an action without an op")
    return {"action_type": "graph_edit_plan", "value": {"steps": steps}}


def plan_prompt_messages(row: Mapping[str, object]) -> list[dict[str, str]]:
    ir = build_constraint_ir(row)
    if ir.task_mode != "edit":
        raise ValueError(f"GraphEditDSL plan prompt received task_mode={ir.task_mode!r}")
    user_payload = {
        "constraint_ir": ir.to_dict(),
        "response_schema": {
            "action_type": "graph_edit_plan",
            "value": {"steps": "one or two executable graph_edit_dsl actions"},
        },
    }
    return [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(user_payload, sort_keys=True, separators=(",", ":")),
        },
    ]
