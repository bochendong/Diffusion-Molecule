#!/usr/bin/env python3
"""Target-hidden protocol for direct constraint-repair trajectory generation."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence


PROTOCOL = "common_llm_direct_constraint_repair_v1"
SYSTEM_PROMPT = (
    "You are the planning controller of a unified molecular constraint agent. "
    "Plan which requested properties a source-preserving edit tool should repair, in order. "
    "The executor will launch independent trajectories and observe train-only verifier feedback "
    "after each edit. Do not propose or rank molecules. Return exactly one JSON object with "
    "action_type=constraint_repair_plan and no prose or markdown."
)


def stable_value(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big")


def prompt_messages(
    row: Mapping[str, object],
    properties: Sequence[str],
    margins: Mapping[str, float],
    *,
    max_steps: int,
) -> list[dict[str, str]]:
    payload = {
        "protocol": PROTOCOL,
        "source_smiles": str(row["source_smiles"]),
        "task_id": str(row.get("_uca_task_id", row.get("external_task_id", ""))),
        "constraints": [
            {
                "property": prop,
                "current_train_verifier_margin": round(float(margins[prop]), 6),
            }
            for prop in properties
        ],
        "available_tool": {
            "action_type": "constraint_repair_plan",
            "executor": "source-preserving train-derived delta trajectory",
            "feedback_after_each_step": "updated train-only verifier margins",
        },
        "response_schema": {
            "action_type": "constraint_repair_plan",
            "value": {
                "property_order": list(properties),
                "max_steps": int(max_steps),
            },
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, sort_keys=True, separators=(",", ":"))},
    ]


def plan_payload(property_order: Sequence[str], *, max_steps: int) -> dict[str, object]:
    return {
        "action_type": "constraint_repair_plan",
        "value": {
            "property_order": [str(prop) for prop in property_order],
            "max_steps": int(max_steps),
        },
    }


def validate_plan(
    payload: object,
    *,
    properties: Sequence[str],
    max_steps: int,
) -> tuple[str, ...] | None:
    if not isinstance(payload, Mapping) or payload.get("action_type") != "constraint_repair_plan":
        return None
    value = payload.get("value")
    if not isinstance(value, Mapping):
        return None
    try:
        returned_max_steps = int(value.get("max_steps", -1))
    except (TypeError, ValueError):
        return None
    if returned_max_steps != int(max_steps):
        return None
    order = value.get("property_order")
    if not isinstance(order, list):
        return None
    normalized = tuple(str(prop).strip().lower() for prop in order)
    expected = tuple(str(prop).strip().lower() for prop in properties)
    if len(normalized) != len(expected) or len(set(normalized)) != len(normalized):
        return None
    return normalized if set(normalized) == set(expected) else None
