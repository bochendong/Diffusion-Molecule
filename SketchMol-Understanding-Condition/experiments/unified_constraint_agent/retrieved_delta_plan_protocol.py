#!/usr/bin/env python3
"""Typed common-LLM contract for an executable RetrievedDeltaEdit action."""

from __future__ import annotations

import json
from typing import Mapping

from molecular_constraint_ir import build_constraint_ir


SYSTEM_PROMPT = (
    "You are a unified molecular planning agent. Read the constraint IR and select one executable "
    "RetrievedDeltaEdit action. The action replaces the query side chain with a train-retrieved target "
    "side chain while leaving the query core untouched. Prefer an action that satisfies every requested "
    "property and preserves source similarity. Return exactly one JSON object with "
    "action_type=retrieved_delta_edit. Do not add prose or markdown."
)


def prompt_messages(row: Mapping[str, object]) -> list[dict[str, str]]:
    ir = build_constraint_ir(row)
    if ir.task_mode != "edit":
        raise ValueError(f"RetrievedDeltaEdit prompt received task_mode={ir.task_mode!r}")
    payload = {
        "constraint_ir": ir.to_dict(),
        "response_schema": {
            "action_type": "retrieved_delta_edit",
            "value": {
                "op": "replace_side_chain",
                "query_variable": "one-cut side chain from the query source",
                "retrieved_source_variable": "matched train-source side chain",
                "target_variable": "matched train-target side chain",
            },
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, sort_keys=True, separators=(",", ":")),
        },
    ]


def action_payload(candidate: Mapping[str, object]) -> dict[str, object]:
    query_variable = str(candidate.get("delta_query_variable", "") or "").strip()
    source_variable = str(candidate.get("delta_source_variable", "") or "").strip()
    target_variable = str(candidate.get("delta_target_variable", "") or "").strip()
    if not query_variable or not source_variable or not target_variable:
        raise ValueError("RetrievedDeltaEdit action is missing a one-cut fragment")
    return {
        "action_type": "retrieved_delta_edit",
        "value": {
            "op": "replace_side_chain",
            "query_variable": query_variable,
            "retrieved_source_variable": source_variable,
            "target_variable": target_variable,
        },
    }
