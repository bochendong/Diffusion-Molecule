#!/usr/bin/env python3
"""Shared P32 state, action-support, reward, and rollout helpers."""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
UCA_DIR = SCRIPT_DIR.parent / "unified_constraint_agent"
P31_DIR = SCRIPT_DIR.parent / "p31_1_frontier_online_rloo"
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
for path in (UCA_DIR, P31_DIR, P25_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common_llm_tool_policy as common_policy  # noqa: E402
import train_common_llm_tool_policy_grpo as common_rl  # noqa: E402
import train_p23_joint_grpo as p25  # noqa: E402
from rloo_math import scalar_reward  # noqa: E402


PROTOCOL = "p32_unified_graph_repair_rl_v1"


@dataclass(frozen=True)
class CandidateFeedback:
    reward: float
    valid: bool
    strict_success: bool
    relaxed_success: bool
    details: dict[str, object]


@dataclass(frozen=True)
class RepairAction:
    payload: dict[str, object]
    next_smiles: str
    terminal: bool
    kind: str


@dataclass(frozen=True)
class SupportBundle:
    support: common_rl.PolicyStateSupport
    actions: tuple[RepairAction, ...]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def benchmark_response(smiles: str, mode: str) -> str:
    return p25.protocol.response(smiles, mode)


def score_smiles(record: Mapping[str, object], smiles: str) -> CandidateFeedback:
    row = record["benchmark_row"]
    if not isinstance(row, Mapping):
        raise TypeError("P32 record is missing benchmark_row")
    mode = str(record["task_mode"])
    try:
        raw = benchmark_response(smiles, mode)
    except ValueError:
        raw = ""
    _legacy_reward, details = p25.reward_response(row, raw)
    channels = {"unused": 0.0}
    reward = scalar_reward(channels, details, mode)
    return CandidateFeedback(
        reward=float(reward),
        valid=bool(details.get("valid")),
        strict_success=bool(details.get("strict")),
        relaxed_success=bool(details.get("relaxed")),
        details=dict(details),
    )


def initial_smiles(record: Mapping[str, object]) -> str:
    return str(record.get("initial_smiles", "") or "").strip()


def constraint_ir(record: Mapping[str, object]) -> dict[str, object]:
    value = record.get("constraint_ir", {})
    if not isinstance(value, Mapping):
        raise TypeError("P32 constraint_ir must be an object")
    return dict(value)


def prompt_messages(
    record: Mapping[str, object],
    *,
    current_smiles: str,
    history: Sequence[Mapping[str, object]],
    step_index: int,
    max_steps: int,
) -> list[dict[str, str]]:
    ir = constraint_ir(record)
    mode = str(record["task_mode"])
    source = str(ir.get("source_smiles", "") or "")
    messages = common_policy.policy_prompt_messages(
        ir,
        current_smiles=current_smiles,
        original_source_smiles=source,
        previous_steps=history,
        step_index=step_index,
        max_steps=max_steps,
    )
    system = dict(messages[0])
    system["content"] = (
        "You are one unified molecular graph-repair policy for BUILD and MODIFY. "
        "Return exactly one JSON tool call from the supplied executable support. "
        "BUILD repairs a frozen direct proposal; MODIFY begins from the source and may "
        "accept the frozen direct proposal. Use verifier feedback from previous steps."
    )
    user_payload = json.loads(messages[1]["content"])
    user_payload["protocol"] = PROTOCOL
    user_payload["task_mode"] = mode
    user_payload["environment"]["direct_proposal_smiles"] = str(
        record.get("direct_smiles", "") or ""
    )
    return [
        system,
        {"role": "user", "content": json.dumps(user_payload, sort_keys=True, separators=(",", ":"))},
    ]


def executable_actions(
    record: Mapping[str, object],
    current_smiles: str,
    *,
    step_index: int,
    max_actions: int,
    site_limit: int,
) -> list[RepairAction]:
    graph_limit = max(1, int(max_actions) - 2)
    graph_actions = common_policy.executable_grammar_actions(
        current_smiles,
        site_limit=int(site_limit),
        max_actions=graph_limit,
        include_stop=False,
    )
    output = [
        RepairAction(item.payload, item.next_smiles, item.terminal, "graph_edit")
        for item in graph_actions
    ]
    seen = {common_policy.graph_policy_module().unified.safe_canonical_smiles(item.next_smiles) for item in output}
    direct = str(record.get("direct_smiles", "") or "").strip()
    direct_canonical = common_policy.graph_policy_module().unified.safe_canonical_smiles(direct)
    current_canonical = common_policy.graph_policy_module().unified.safe_canonical_smiles(current_smiles)
    if step_index == 0 and direct_canonical and direct_canonical not in seen and direct_canonical != current_canonical:
        output.append(RepairAction(
            {
                "action_type": "accept_direct_proposal",
                "value": {"proposal": direct_canonical},
            },
            direct_canonical,
            False,
            "direct_proposal",
        ))
        seen.add(direct_canonical)
    if current_canonical:
        output.append(RepairAction(
            {"action_type": "stop", "value": {"reason": "keep_current_molecule"}},
            current_canonical,
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
        record, current_smiles, step_index=step_index,
        max_actions=max_actions, site_limit=site_limit,
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
    payloads = [item.payload for item in actions]
    scores = common_rl.inference_action_scores(
        model,
        tokenizer,
        messages,
        payloads,
        max_length=int(max_length),
        batch_size=int(score_batch_size),
    )
    feedback = [score_smiles(record, item.next_smiles) for item in actions]
    support = common_rl.PolicyStateSupport(
        origin=str(record["task_mode"]),
        example_id=str(record.get("example_id", "")),
        prompt_messages=messages,
        candidate_payloads=payloads,
        candidate_smiles=[item.next_smiles for item in actions],
        action_scores=scores,
        feedback=feedback,
    )
    return SupportBundle(support, tuple(actions))


def sample_action_index(scores: Sequence[float], *, temperature: float, seed: int) -> int:
    probabilities, _advantages, _weights = common_rl.exact_action_distribution(
        scores, [0.0] * len(scores), temperature=temperature
    )
    rng = random.Random(int(seed))
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw <= cumulative:
            return index
    return max(0, len(probabilities) - 1)


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
    current = initial_smiles(record)
    history: list[dict[str, object]] = []
    supports = []
    for step_index in range(int(max_steps)):
        bundle = support_bundle(
            model, tokenizer, record,
            current_smiles=current, history=history, step_index=step_index,
            max_steps=max_steps, max_actions=max_actions, site_limit=site_limit,
            max_length=max_length, score_batch_size=score_batch_size,
        )
        if bundle is None:
            break
        supports.append(bundle.support)
        selected_index = sample_action_index(
            bundle.support.action_scores,
            temperature=temperature,
            seed=int(seed) + step_index,
        )
        action = bundle.actions[selected_index]
        feedback = bundle.support.feedback[selected_index]
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
    current = initial_smiles(record)
    history: list[dict[str, object]] = []
    trace = []
    for step_index in range(int(max_steps)):
        bundle = support_bundle(
            model, tokenizer, record,
            current_smiles=current, history=history, step_index=step_index,
            max_steps=max_steps, max_actions=max_actions, site_limit=site_limit,
            max_length=max_length, score_batch_size=score_batch_size,
        )
        if bundle is None:
            break
        selected_index = max(
            range(len(bundle.support.action_scores)),
            key=lambda index: float(bundle.support.action_scores[index]),
        )
        action = bundle.actions[selected_index]
        feedback = bundle.support.feedback[selected_index]
        current = action.next_smiles
        row = {
            "step": step_index,
            "action": action.payload,
            "kind": action.kind,
            "smiles": current,
            "score": float(bundle.support.action_scores[selected_index]),
            "reward": feedback.reward,
            "strict": feedback.strict_success,
        }
        trace.append(row)
        history.append({"tool_call": action.payload, "result_smiles": current, "observation": row})
        if action.terminal:
            break
    return current, trace


def mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / max(len(values), 1)


def aggregate_records(rows: Sequence[Mapping[str, object]], details_key: str) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["bucket"]), []).append(row)
    buckets = {}
    for bucket, items in sorted(grouped.items()):
        buckets[bucket] = {
            "rows": len(items),
            "strict_rate": mean(bool(item[details_key]["strict"]) for item in items),
            "relaxed_rate": mean(bool(item[details_key]["relaxed"]) for item in items),
            "valid_rate": mean(bool(item[details_key]["valid"]) for item in items),
        }
    return {
        "rows": len(rows),
        "strict_macro": mean(value["strict_rate"] for value in buckets.values()),
        "relaxed_macro": mean(value["relaxed_rate"] for value in buckets.values()),
        "valid_macro": mean(value["valid_rate"] for value in buckets.values()),
        "buckets": buckets,
    }
