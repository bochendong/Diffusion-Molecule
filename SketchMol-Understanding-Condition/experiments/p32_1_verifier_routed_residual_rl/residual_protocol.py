#!/usr/bin/env python3
"""Verifier-routed residual state, support, and rollout helpers for P32.1."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P32_DIR = SCRIPT_DIR.parent / "p32_unified_graph_repair_rl"
UCA_DIR = SCRIPT_DIR.parent / "unified_constraint_agent"
for path in (P32_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import graph_repair_protocol as p32  # noqa: E402
import common_llm_tool_policy as common_policy  # noqa: E402
import train_common_llm_tool_policy_grpo as common_rl  # noqa: E402


PROTOCOL = "p32_1_verifier_routed_residual_rl_v1"
CandidateFeedback = p32.CandidateFeedback
RepairAction = p32.RepairAction
SupportBundle = p32.SupportBundle
read_jsonl = p32.read_jsonl
write_jsonl = p32.write_jsonl
score_smiles = p32.score_smiles
mean = p32.mean
aggregate_records = p32.aggregate_records


def direct_smiles(record: Mapping[str, object]) -> str:
    return str(record.get("direct_smiles", "") or "").strip()


def direct_feedback(record: Mapping[str, object]) -> CandidateFeedback:
    return score_smiles(record, direct_smiles(record))


def hard_accept_direct(record: Mapping[str, object]) -> bool:
    return direct_feedback(record).strict_success


def initial_smiles(record: Mapping[str, object]) -> str:
    return direct_smiles(record) or "C"


def prompt_messages(
    record: Mapping[str, object],
    *,
    current_smiles: str,
    history: Sequence[Mapping[str, object]],
    step_index: int,
    max_steps: int,
) -> list[dict[str, str]]:
    messages = p32.prompt_messages(
        record,
        current_smiles=current_smiles,
        history=history,
        step_index=step_index,
        max_steps=max_steps,
    )
    messages[0] = {
        "role": "system",
        "content": (
            "You are one shared residual molecular graph-repair policy for BUILD and MODIFY. "
            "The frozen direct proposal failed the target-blind verifier. Return exactly one "
            "JSON tool call from the executable support. Prefer stop when no safe improvement "
            "is supported; never invent an action outside the supplied grammar."
        ),
    }
    payload = json.loads(messages[1]["content"])
    initial = direct_feedback(record)
    payload["protocol"] = PROTOCOL
    payload["routing"] = "strict direct proposals were already hard-accepted"
    payload["environment"]["initial_verifier_feedback"] = {
        "reward": initial.reward,
        "valid": initial.valid,
        "strict_success": initial.strict_success,
        "relaxed_success": initial.relaxed_success,
        "details": initial.details,
    }
    messages[1] = {
        "role": "user",
        "content": json.dumps(payload, sort_keys=True, separators=(",", ":")),
    }
    return messages


def executable_actions(
    record: Mapping[str, object],
    current_smiles: str,
    *,
    step_index: int,
    max_actions: int,
    site_limit: int,
) -> list[RepairAction]:
    del record, step_index
    canonical = common_policy.graph_policy_module().unified.safe_canonical_smiles(current_smiles)
    graph_actions = common_policy.executable_grammar_actions(
        current_smiles,
        site_limit=int(site_limit),
        max_actions=max(1, int(max_actions) - 1),
        include_stop=False,
    )
    output = [
        RepairAction(item.payload, item.next_smiles, item.terminal, "graph_edit")
        for item in graph_actions
    ]
    if canonical:
        output.append(RepairAction(
            {"action_type": "stop", "value": {"reason": "keep_current_proposal"}},
            canonical,
            True,
            "stop",
        ))
    return output[: max(1, int(max_actions))]


def support_bundle(
    model: object,
    tokenizer: object,
    record: Mapping[str, object],
    *,
    current_smiles: str,
    history: Sequence[Mapping[str, object]],
    step_index: int,
    max_steps: int,
    max_actions: int,
    site_limit: int,
    max_length: int,
    score_batch_size: int,
) -> SupportBundle | None:
    actions = executable_actions(
        record,
        current_smiles,
        step_index=step_index,
        max_actions=max_actions,
        site_limit=site_limit,
    )
    if not actions:
        return None
    messages = prompt_messages(
        record,
        current_smiles=current_smiles,
        history=history,
        step_index=step_index,
        max_steps=max_steps,
    )
    payloads = [action.payload for action in actions]
    scores = common_rl.inference_action_scores(
        model,
        tokenizer,
        messages,
        payloads,
        max_length=int(max_length),
        batch_size=int(score_batch_size),
    )
    feedback = [score_smiles(record, action.next_smiles) for action in actions]
    support = common_rl.PolicyStateSupport(
        origin=str(record["task_mode"]),
        example_id=str(record.get("example_id", "")),
        prompt_messages=messages,
        candidate_payloads=payloads,
        candidate_smiles=[action.next_smiles for action in actions],
        action_scores=scores,
        feedback=feedback,
    )
    return SupportBundle(support, tuple(actions))


def online_supports(
    model: object,
    tokenizer: object,
    record: Mapping[str, object],
    *,
    max_steps: int,
    max_actions: int,
    site_limit: int,
    max_length: int,
    score_batch_size: int,
    temperature: float,
    seed: int,
) -> tuple[list[common_rl.PolicyStateSupport], str]:
    if hard_accept_direct(record):
        return [], direct_smiles(record)
    current = initial_smiles(record)
    history: list[dict[str, object]] = []
    supports = []
    for step_index in range(int(max_steps)):
        bundle = support_bundle(
            model,
            tokenizer,
            record,
            current_smiles=current,
            history=history,
            step_index=step_index,
            max_steps=max_steps,
            max_actions=max_actions,
            site_limit=site_limit,
            max_length=max_length,
            score_batch_size=score_batch_size,
        )
        if bundle is None:
            break
        supports.append(bundle.support)
        selected = p32.sample_action_index(
            bundle.support.action_scores,
            temperature=temperature,
            seed=int(seed) + step_index,
        )
        action = bundle.actions[selected]
        feedback = bundle.support.feedback[selected]
        current = action.next_smiles
        history.append({
            "tool_call": action.payload,
            "result_smiles": current,
            "observation": {
                "reward": feedback.reward,
                "valid": feedback.valid,
                "strict_success": feedback.strict_success,
                "details": feedback.details,
            },
        })
        if action.terminal:
            break
    return supports, current


def greedy_rollout(
    model: object,
    tokenizer: object,
    record: Mapping[str, object],
    *,
    max_steps: int,
    max_actions: int,
    site_limit: int,
    max_length: int,
    score_batch_size: int,
) -> tuple[str, list[dict[str, object]]]:
    if hard_accept_direct(record):
        return direct_smiles(record), [{
            "step": -1,
            "kind": "hard_accept_direct",
            "smiles": direct_smiles(record),
            "strict": True,
        }]
    current = initial_smiles(record)
    history: list[dict[str, object]] = []
    trace = []
    for step_index in range(int(max_steps)):
        bundle = support_bundle(
            model,
            tokenizer,
            record,
            current_smiles=current,
            history=history,
            step_index=step_index,
            max_steps=max_steps,
            max_actions=max_actions,
            site_limit=site_limit,
            max_length=max_length,
            score_batch_size=score_batch_size,
        )
        if bundle is None:
            break
        selected = max(
            range(len(bundle.support.action_scores)),
            key=lambda index: float(bundle.support.action_scores[index]),
        )
        action = bundle.actions[selected]
        feedback = bundle.support.feedback[selected]
        current = action.next_smiles
        row = {
            "step": step_index,
            "action": action.payload,
            "kind": action.kind,
            "smiles": current,
            "score": float(bundle.support.action_scores[selected]),
            "reward": feedback.reward,
            "strict": feedback.strict_success,
        }
        trace.append(row)
        history.append({"tool_call": action.payload, "result_smiles": current, "observation": row})
        if action.terminal:
            break
    return current, trace
