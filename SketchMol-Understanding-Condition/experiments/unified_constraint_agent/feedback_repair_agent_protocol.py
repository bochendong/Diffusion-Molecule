#!/usr/bin/env python3
"""Target-hidden protocol for event-driven molecular constraint repair."""

from __future__ import annotations

import json
from typing import Mapping, Sequence


PROTOCOL = "common_llm_feedback_repair_controller_v1"
SYSTEM_PROMPT = (
    "You are the event-driven controller of one unified molecular constraint agent. "
    "Read train-only verifier margins, train-derived tool support, and the previous "
    "commit or rollback event. Choose exactly one next tool action: repair one requested "
    "property or stop after at least one committed edit. Do not propose molecules, rank "
    "molecules, or use evaluation targets. Return exactly one JSON object and no prose."
)


def repair_action(prop: str) -> dict[str, object]:
    return {"action_type": "repair", "value": {"property": str(prop)}}


def stop_action(reason: str = "constraints_satisfied_or_no_safe_support") -> dict[str, object]:
    return {"action_type": "stop", "value": {"reason": str(reason)}}


def candidate_actions(
    properties: Sequence[str],
    support: Mapping[str, Mapping[str, object]],
    *,
    committed_edits: int,
) -> list[dict[str, object]]:
    output = [
        repair_action(prop)
        for prop in properties
        if int(dict(support.get(prop, {})).get("executable_actions", 0)) > 0
    ]
    if int(committed_edits) > 0:
        output.append(stop_action())
    return output


def prompt_messages(
    row: Mapping[str, object],
    properties: Sequence[str],
    margins: Mapping[str, float],
    support: Mapping[str, Mapping[str, object]],
    *,
    current_smiles: str,
    committed_edits: int,
    proposal_count: int,
    max_committed_edits: int,
    max_proposals: int,
    previous_event: Mapping[str, object] | None,
) -> list[dict[str, str]]:
    payload = {
        "protocol": PROTOCOL,
        "task_id": str(row.get("_uca_task_id", row.get("external_task_id", ""))),
        "source_smiles": str(row.get("source_smiles", "")),
        "current_smiles": str(current_smiles),
        "constraints": [
            {
                "property": str(prop),
                "current_train_verifier_margin": round(float(margins[prop]), 6),
                "tool_support": dict(support.get(prop, {})),
            }
            for prop in properties
        ],
        "trajectory": {
            "committed_edits": int(committed_edits),
            "proposal_count": int(proposal_count),
            "max_committed_edits": int(max_committed_edits),
            "max_proposals": int(max_proposals),
            "previous_event": dict(previous_event) if previous_event is not None else None,
        },
        "available_actions": candidate_actions(
            properties, support, committed_edits=int(committed_edits)
        ),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, sort_keys=True, separators=(",", ":"))},
    ]


def validate_action(
    payload: object,
    *,
    properties: Sequence[str],
    allow_stop: bool,
) -> tuple[str, str | None] | None:
    if not isinstance(payload, Mapping):
        return None
    action_type = str(payload.get("action_type", ""))
    value = payload.get("value")
    if not isinstance(value, Mapping):
        return None
    if action_type == "repair":
        prop = str(value.get("property", "")).strip().lower()
        expected = {str(item).strip().lower() for item in properties}
        return ("repair", prop) if prop in expected else None
    if action_type == "stop" and allow_stop:
        return ("stop", None)
    return None


def state_contains_evaluation_target(messages: Sequence[Mapping[str, str]]) -> bool:
    serialized = json.dumps(list(messages), sort_keys=True).lower()
    return "target_smiles" in serialized or "external_target_" in serialized
