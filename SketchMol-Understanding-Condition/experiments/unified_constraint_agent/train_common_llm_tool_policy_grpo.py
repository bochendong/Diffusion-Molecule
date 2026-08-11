#!/usr/bin/env python3
"""Train the common LLM as a closed-loop typed molecular tool policy.

Unlike the earlier preference experiments, this trainer does not consume a
ranked molecular candidate pool or mine strict-positive pairs. It executes
GraphEditDSL calls from the current policy, returns constraint feedback, and
supports either sampled trajectory GRPO or exact action-value policy updates
over the complete typed support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import common_llm_tool_policy as policy  # noqa: E402
import evaluate_common_llm_constrained_actions as constrained  # noqa: E402
import train_common_llm_preference as common_train  # noqa: E402


@dataclass
class PolicyDecision:
    prompt_messages: list[dict[str, str]]
    candidate_payloads: list[dict[str, object]]
    selected_index: int


@dataclass
class PolicyTrajectory:
    origin: str
    example_id: str
    final_smiles: str
    decisions: list[PolicyDecision]
    feedback: policy.PolicyFeedback


@dataclass
class PolicyStateSupport:
    origin: str
    example_id: str
    prompt_messages: list[dict[str, str]]
    candidate_payloads: list[dict[str, object]]
    candidate_smiles: list[str]
    action_scores: list[float]
    feedback: list[policy.PolicyFeedback]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--input-adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--rollout-origins", default="table1,mumo")
    parser.add_argument("--max-train-per-origin", type=int, default=16)
    parser.add_argument("--max-validation-per-origin", type=int, default=8)
    parser.add_argument("--rollouts-per-condition", type=int, default=4)
    parser.add_argument(
        "--policy-update",
        choices=("sampled_trajectory", "exact_action_value"),
        default="sampled_trajectory",
    )
    parser.add_argument("--paths-per-condition", type=int, default=1)
    parser.add_argument("--validation-rollouts", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--site-limit", type=int, default=24)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--table1-similarity-threshold", type=float, default=0.65)
    parser.add_argument("--mumo-similarity-threshold", type=float, default=0.40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--anchor-sft-weight", type=float, default=0.10)
    parser.add_argument("--anchor-max-per-origin", type=int, default=64)
    parser.add_argument("--retention-max-per-origin", type=int, default=32)
    parser.add_argument("--retention-max-regression", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1707)
    parser.add_argument("--logging-steps", type=int, default=2)
    return parser.parse_args(argv)


def requested_origins(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def unique_records(
    rows: Sequence[Mapping[str, object]],
    *,
    origins: Sequence[str],
    max_per_origin: int,
    seed: int,
) -> list[Mapping[str, object]]:
    requested = set(origins)
    grouped: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for index, row in enumerate(rows):
        origin = str(row.get("origin", "") or "").strip()
        if origin not in requested or str(row.get("task_mode", "") or "") != "edit":
            continue
        identity = str(row.get("example_id", "") or f"{origin}:{index}")
        grouped[origin].setdefault(identity, row)
    missing = sorted(requested - set(grouped))
    if missing:
        raise ValueError(f"Tool-policy split is missing edit origins: {missing}")
    output: list[Mapping[str, object]] = []
    for offset, origin in enumerate(origins):
        records = list(grouped[origin].values())
        random.Random(int(seed) + offset).shuffle(records)
        output.extend(records[: max(1, int(max_per_origin))])
    random.Random(int(seed)).shuffle(output)
    return output


def similarity_threshold(origin: str, args: argparse.Namespace) -> float:
    return (
        float(args.mumo_similarity_threshold)
        if str(origin) == "mumo"
        else float(args.table1_similarity_threshold)
    )


def _decision_signature(decision: PolicyDecision) -> str:
    return hashlib.sha256(
        json.dumps(
            [decision.prompt_messages, decision.candidate_payloads[decision.selected_index]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def inference_action_scores(
    model: object,
    tokenizer: object,
    prompt_messages: Sequence[Mapping[str, object]],
    payloads: Sequence[Mapping[str, object]],
    *,
    max_length: int,
    batch_size: int,
) -> list[float]:
    encoded = [
        constrained.encoded_action(tokenizer, prompt_messages, payload, max_length=int(max_length))
        for payload in payloads
    ]
    return constrained.score_encoded_actions(
        model,
        tokenizer,
        encoded,
        batch_size=int(batch_size),
    )


def sample_index(scores: Sequence[float], *, temperature: float, generator: object) -> int:
    import torch

    logits = torch.tensor(scores, dtype=torch.float32) / max(float(temperature), 1e-6)
    probabilities = torch.softmax(logits, dim=0)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def rollout_group(
    model: object,
    tokenizer: object,
    record: Mapping[str, object],
    *,
    args: argparse.Namespace,
    rollout_count: int,
    seed: int,
) -> list[PolicyTrajectory]:
    import torch

    ir = policy.constraint_ir(record)
    origin = str(record.get("origin", "") or "unknown")
    example_id = str(record.get("example_id", "") or "")
    source = str(ir.get("source_smiles", "") or "").strip()
    if not source:
        raise ValueError(f"Edit policy row has no source SMILES: {example_id}")
    cache: dict[str, list[float]] = {}
    trajectories: list[PolicyTrajectory] = []
    model.eval()
    for rollout_index in range(int(rollout_count)):
        current = source
        history: list[dict[str, object]] = []
        decisions: list[PolicyDecision] = []
        final_feedback = policy.score_policy_state(
            ir,
            original_source_smiles=source,
            candidate_smiles=current,
            source_similarity_threshold=similarity_threshold(origin, args),
            step_count=0,
        )
        generator = torch.Generator().manual_seed(int(seed) + rollout_index)
        for step_index in range(int(args.max_steps)):
            actions = policy.executable_grammar_actions(
                current,
                site_limit=int(args.site_limit),
                max_actions=int(args.max_actions),
                include_stop=step_index > 0,
            )
            if not actions:
                break
            messages = policy.policy_prompt_messages(
                ir,
                current_smiles=current,
                original_source_smiles=source,
                previous_steps=history,
                step_index=step_index,
                max_steps=int(args.max_steps),
            )
            payloads = [action.payload for action in actions]
            signature = hashlib.sha256(
                json.dumps([messages, payloads], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if signature not in cache:
                cache[signature] = inference_action_scores(
                    model,
                    tokenizer,
                    messages,
                    payloads,
                    max_length=int(args.max_length),
                    batch_size=int(args.score_batch_size),
                )
            selected_index = sample_index(
                cache[signature],
                temperature=float(args.temperature),
                generator=generator,
            )
            selected = actions[selected_index]
            current = selected.next_smiles
            final_feedback = policy.score_policy_state(
                ir,
                original_source_smiles=source,
                candidate_smiles=current,
                source_similarity_threshold=similarity_threshold(origin, args),
                step_count=step_index + int(not selected.terminal),
            )
            missing_feedback = [
                item.property
                for item in final_feedback.outcomes
                if item.source_value is None or item.candidate_value is None
            ]
            if missing_feedback:
                raise RuntimeError(
                    f"Incomplete official feedback for {example_id}: {sorted(set(missing_feedback))}"
                )
            decisions.append(PolicyDecision(messages, payloads, selected_index))
            history.append(
                {
                    "tool_call": selected.payload,
                    "result_smiles": current,
                    "observation": final_feedback.observation(),
                }
            )
            if selected.terminal:
                break
        trajectories.append(
            PolicyTrajectory(origin, example_id, current, decisions, final_feedback)
        )
    return trajectories


def exact_action_distribution(
    scores: Sequence[float],
    rewards: Sequence[float],
    *,
    temperature: float,
    clip: float = 3.0,
) -> tuple[list[float], list[float], list[float]]:
    """Return categorical probabilities, centered advantages, and policy weights.

    The executable tool support is small and discrete, so v2 can replace a
    Monte-Carlo group estimate with the exact expectation over every action.
    Centering under the policy distribution keeps the resulting score-function
    weights zero-sum even when advantages are clipped.
    """
    if len(scores) != len(rewards):
        raise ValueError("Action scores and rewards must have the same length")
    if not scores:
        return [], [], []
    inverse_temperature = 1.0 / max(float(temperature), 1e-6)
    logits = [float(item) * inverse_temperature for item in scores]
    maximum = max(logits)
    unnormalized = [math.exp(item - maximum) for item in logits]
    denominator = sum(unnormalized)
    probabilities = [item / denominator for item in unnormalized]
    expected_reward = sum(
        probability * float(reward) for probability, reward in zip(probabilities, rewards)
    )
    variance = sum(
        probability * (float(reward) - expected_reward) ** 2
        for probability, reward in zip(probabilities, rewards)
    )
    scale = max(variance**0.5, 1e-6)
    advantages = [
        max(-float(clip), min(float(clip), (float(reward) - expected_reward) / scale))
        for reward in rewards
    ]
    clipped_center = sum(
        probability * advantage for probability, advantage in zip(probabilities, advantages)
    )
    advantages = [advantage - clipped_center for advantage in advantages]
    weights = [
        probability * advantage for probability, advantage in zip(probabilities, advantages)
    ]
    return probabilities, advantages, weights


def evaluate_action_support(
    model: object,
    tokenizer: object,
    record: Mapping[str, object],
    *,
    current_smiles: str,
    original_source_smiles: str,
    history: Sequence[Mapping[str, object]],
    step_index: int,
    args: argparse.Namespace,
) -> PolicyStateSupport | None:
    ir = policy.constraint_ir(record)
    origin = str(record.get("origin", "") or "unknown")
    example_id = str(record.get("example_id", "") or "")
    actions = policy.executable_grammar_actions(
        current_smiles,
        site_limit=int(args.site_limit),
        max_actions=int(args.max_actions),
        include_stop=int(step_index) > 0,
    )
    if not actions:
        return None
    messages = policy.policy_prompt_messages(
        ir,
        current_smiles=current_smiles,
        original_source_smiles=original_source_smiles,
        previous_steps=history,
        step_index=int(step_index),
        max_steps=int(args.max_steps),
    )
    payloads = [action.payload for action in actions]
    action_scores = inference_action_scores(
        model,
        tokenizer,
        messages,
        payloads,
        max_length=int(args.max_length),
        batch_size=int(args.score_batch_size),
    )
    feedback = [
        policy.score_policy_state(
            ir,
            original_source_smiles=original_source_smiles,
            candidate_smiles=action.next_smiles,
            source_similarity_threshold=similarity_threshold(origin, args),
            step_count=int(step_index) + int(not action.terminal),
        )
        for action in actions
    ]
    missing_feedback = sorted(
        {
            outcome.property
            for item in feedback
            for outcome in item.outcomes
            if outcome.source_value is None or outcome.candidate_value is None
        }
    )
    if missing_feedback:
        raise RuntimeError(
            f"Incomplete official action-value feedback for {example_id}: {missing_feedback}"
        )
    return PolicyStateSupport(
        origin=origin,
        example_id=example_id,
        prompt_messages=messages,
        candidate_payloads=payloads,
        candidate_smiles=[action.next_smiles for action in actions],
        action_scores=action_scores,
        feedback=feedback,
    )


def rollout_exact_action_supports(
    model: object,
    tokenizer: object,
    record: Mapping[str, object],
    *,
    args: argparse.Namespace,
    path_count: int,
    seed: int,
) -> tuple[list[PolicyStateSupport], list[PolicyTrajectory]]:
    """Visit policy states while evaluating every executable action exactly."""
    import torch

    ir = policy.constraint_ir(record)
    origin = str(record.get("origin", "") or "unknown")
    example_id = str(record.get("example_id", "") or "")
    source = str(ir.get("source_smiles", "") or "").strip()
    if not source:
        raise ValueError(f"Edit policy row has no source SMILES: {example_id}")
    support_cache: dict[str, PolicyStateSupport] = {}
    trajectories: list[PolicyTrajectory] = []
    model.eval()
    for path_index in range(max(1, int(path_count))):
        current = source
        history: list[dict[str, object]] = []
        decisions: list[PolicyDecision] = []
        final_feedback = policy.score_policy_state(
            ir,
            original_source_smiles=source,
            candidate_smiles=current,
            source_similarity_threshold=similarity_threshold(origin, args),
            step_count=0,
        )
        generator = torch.Generator().manual_seed(int(seed) + path_index)
        for step_index in range(int(args.max_steps)):
            actions = policy.executable_grammar_actions(
                current,
                site_limit=int(args.site_limit),
                max_actions=int(args.max_actions),
                include_stop=step_index > 0,
            )
            if not actions:
                break
            messages = policy.policy_prompt_messages(
                ir,
                current_smiles=current,
                original_source_smiles=source,
                previous_steps=history,
                step_index=step_index,
                max_steps=int(args.max_steps),
            )
            payloads = [action.payload for action in actions]
            signature = hashlib.sha256(
                json.dumps([messages, payloads], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            support = support_cache.get(signature)
            if support is None:
                support = evaluate_action_support(
                    model,
                    tokenizer,
                    record,
                    current_smiles=current,
                    original_source_smiles=source,
                    history=history,
                    step_index=step_index,
                    args=args,
                )
                if support is None:
                    break
                support_cache[signature] = support
            selected_index = sample_index(
                support.action_scores,
                temperature=float(args.temperature),
                generator=generator,
            )
            selected = actions[selected_index]
            current = selected.next_smiles
            final_feedback = support.feedback[selected_index]
            decisions.append(PolicyDecision(messages, payloads, selected_index))
            history.append(
                {
                    "tool_call": selected.payload,
                    "result_smiles": current,
                    "observation": final_feedback.observation(),
                }
            )
            if selected.terminal:
                break
        trajectories.append(
            PolicyTrajectory(origin, example_id, current, decisions, final_feedback)
        )
    return list(support_cache.values()), trajectories


def differentiable_selected_action_score(
    model: object,
    tokenizer: object,
    prompt_messages: Sequence[Mapping[str, object]],
    payload: Mapping[str, object],
    *,
    max_length: int,
):
    import torch

    encoded = constrained.encoded_action(
        tokenizer,
        prompt_messages,
        payload,
        max_length=int(max_length),
    )
    tensors = {
        key: torch.tensor([value], dtype=torch.long, device="cuda")
        for key, value in encoded.items()
    }
    return common_train.sequence_mean_log_probability(model, tensors)[0]


def policy_gradient_backward(
    model: object,
    tokenizer: object,
    trajectories: Sequence[PolicyTrajectory],
    advantages: Sequence[float],
    *,
    args: argparse.Namespace,
) -> tuple[float, int]:
    grouped: dict[str, dict[str, object]] = {}
    total_decisions = 0
    for trajectory, advantage in zip(trajectories, advantages):
        for decision in trajectory.decisions:
            signature = _decision_signature(decision)
            group = grouped.setdefault(
                signature,
                {
                    "decision": decision,
                    "weight": 0.0,
                    "count": 0,
                },
            )
            group["weight"] = float(group["weight"]) + float(advantage)
            group["count"] = int(group["count"]) + 1
            total_decisions += 1
    if not grouped:
        return 0.0, 0
    detached_loss = 0.0
    normalizer = max(total_decisions, 1) * max(int(args.gradient_accumulation), 1)
    for group in grouped.values():
        decision = group["decision"]
        assert isinstance(decision, PolicyDecision)
        weight = float(group["weight"])
        if abs(weight) < 1e-12:
            continue
        selected_payload = decision.candidate_payloads[decision.selected_index]
        selected_log_probability = differentiable_selected_action_score(
            model,
            tokenizer,
            decision.prompt_messages,
            selected_payload,
            max_length=int(args.max_length),
        )
        # Rollout action support is a typed grammar projection. Recompute only
        # the executed autoregressive action probability: token-level LM
        # softmaxes retain the negative alternatives without keeping every
        # candidate sequence graph resident on a 20 GB MIG.
        loss = -weight * selected_log_probability / max(float(args.temperature), 1e-6)
        detached_loss += float(loss.detach())
        (loss / normalizer).backward()
    return detached_loss / max(total_decisions, 1), total_decisions


def exact_action_value_backward(
    model: object,
    tokenizer: object,
    supports: Sequence[PolicyStateSupport],
    *,
    args: argparse.Namespace,
) -> tuple[float, int, dict[str, float]]:
    """Apply an exact score-function update over each typed action support.

    Candidate scores and official rewards are detached. Each action sequence is
    then recomputed separately, keeping peak memory bounded while retaining the
    exact categorical policy-gradient expectation over the complete support.
    """
    if not supports:
        return 0.0, 0, {
            "mean_support_size": 0.0,
            "mean_expected_reward": 0.0,
            "mean_oracle_reward": 0.0,
            "mean_oracle_probability": 0.0,
        }
    detached_loss = 0.0
    action_count = 0
    expected_rewards = []
    oracle_rewards = []
    oracle_probabilities = []
    normalizer = max(len(supports), 1) * max(int(args.gradient_accumulation), 1)
    for support in supports:
        rewards = [item.reward for item in support.feedback]
        probabilities, _advantages, weights = exact_action_distribution(
            support.action_scores,
            rewards,
            temperature=float(args.temperature),
        )
        expected_rewards.append(
            sum(probability * reward for probability, reward in zip(probabilities, rewards))
        )
        oracle_reward = max(rewards)
        oracle_rewards.append(oracle_reward)
        oracle_probabilities.append(
            sum(
                probability
                for probability, reward in zip(probabilities, rewards)
                if abs(float(reward) - float(oracle_reward)) < 1e-9
            )
        )
        for payload, weight in zip(support.candidate_payloads, weights):
            action_count += 1
            if abs(float(weight)) < 1e-12:
                continue
            action_log_probability = differentiable_selected_action_score(
                model,
                tokenizer,
                support.prompt_messages,
                payload,
                max_length=int(args.max_length),
            )
            loss = -float(weight) * action_log_probability / max(float(args.temperature), 1e-6)
            detached_loss += float(loss.detach())
            (loss / normalizer).backward()
    return (
        detached_loss / max(len(supports), 1),
        action_count,
        {
            "mean_support_size": mean(len(item.candidate_payloads) for item in supports),
            "mean_expected_reward": mean(expected_rewards),
            "mean_oracle_reward": mean(oracle_rewards),
            "mean_oracle_probability": mean(oracle_probabilities),
        },
    )


def anchor_examples(
    rows: Sequence[Mapping[str, object]],
    tokenizer: object,
    *,
    max_per_origin: int,
    max_length: int,
    seed: int,
) -> dict[str, list[dict[str, list[int]]]]:
    selected = common_train.task_balanced_replay_rows(
        rows,
        origins=("denovo", "table1", "mumo"),
        max_per_origin=int(max_per_origin),
        seed=int(seed),
    )
    output: dict[str, list[dict[str, list[int]]]] = defaultdict(list)
    for row in selected:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        assistant = messages[-1]
        if not isinstance(assistant, Mapping):
            continue
        try:
            payload = json.loads(str(assistant.get("content", "") or ""))
        except json.JSONDecodeError:
            continue
        output[str(row.get("origin", "") or "unknown")].append(
            constrained.encoded_action(tokenizer, messages[:-1], payload, max_length=int(max_length))
        )
    missing = sorted({"denovo", "table1", "mumo"} - set(output))
    if missing:
        raise ValueError(f"Joint SFT anchor is missing origins: {missing}")
    return dict(output)


def anchor_sft_backward(
    model: object,
    examples: Mapping[str, Sequence[Mapping[str, Sequence[int]]]],
    cursors: dict[str, int],
    *,
    tokenizer: object,
    weight: float,
    gradient_accumulation: int,
) -> float:
    import torch

    losses = []
    for origin in ("denovo", "table1", "mumo"):
        rows = examples[origin]
        item = rows[cursors[origin] % len(rows)]
        cursors[origin] += 1
        tensors = {
            "input_ids": torch.tensor([item["input_ids"]], dtype=torch.long, device="cuda"),
            "attention_mask": torch.tensor([item["attention_mask"]], dtype=torch.long, device="cuda"),
            "labels": torch.tensor([item["labels"]], dtype=torch.long, device="cuda"),
        }
        loss = -common_train.sequence_mean_log_probability(model, tensors).mean()
        (
            float(weight)
            * loss
            / (3.0 * max(int(gradient_accumulation), 1))
        ).backward()
        losses.append(float(loss.detach()))
    return mean(losses)


def trajectory_record(trajectory: PolicyTrajectory) -> dict[str, object]:
    return {
        "origin": trajectory.origin,
        "example_id": trajectory.example_id,
        "final_smiles": trajectory.final_smiles,
        "reward": trajectory.feedback.reward,
        "strict_success": trajectory.feedback.strict_success,
        "property_all_success": trajectory.feedback.property_all_success,
        "property_success_fraction": trajectory.feedback.property_success_fraction,
        "source_similarity": policy.finite_or_none(trajectory.feedback.source_similarity),
        "source_similarity_success": trajectory.feedback.source_similarity_success,
        "steps": [
            {
                "selected_index": decision.selected_index,
                "selected_payload": decision.candidate_payloads[decision.selected_index],
                "action_support_size": len(decision.candidate_payloads),
            }
            for decision in trajectory.decisions
        ],
    }


def aggregate_metrics(trajectories: Sequence[PolicyTrajectory]) -> dict[str, object]:
    grouped: dict[str, list[PolicyTrajectory]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[trajectory.example_id].append(trajectory)
    if not grouped:
        return {"conditions": 0}
    condition_best_rewards = [max(item.feedback.reward for item in group) for group in grouped.values()]
    return {
        "conditions": len(grouped),
        "trajectories": len(trajectories),
        "mean_reward": mean(item.feedback.reward for item in trajectories),
        "mean_best_reward": mean(condition_best_rewards),
        "strict_any_rate": mean(any(item.feedback.strict_success for item in group) for group in grouped.values()),
        "property_all_any_rate": mean(
            any(item.feedback.property_all_success for item in group) for group in grouped.values()
        ),
        "source_similarity_any_rate": mean(
            any(item.feedback.source_similarity_success for item in group) for group in grouped.values()
        ),
        "mean_property_success_fraction": mean(
            item.feedback.property_success_fraction for item in trajectories
        ),
        "mean_steps": mean(len(item.decisions) for item in trajectories),
    }


def action_value_record(support: PolicyStateSupport, *, temperature: float) -> dict[str, object]:
    rewards = [item.reward for item in support.feedback]
    probabilities, _advantages, _weights = exact_action_distribution(
        support.action_scores,
        rewards,
        temperature=float(temperature),
    )
    top_index = max(range(len(support.action_scores)), key=support.action_scores.__getitem__)
    oracle_reward = max(rewards)
    return {
        "origin": support.origin,
        "example_id": support.example_id,
        "support_size": len(support.candidate_payloads),
        "expected_reward": sum(
            probability * reward for probability, reward in zip(probabilities, rewards)
        ),
        "top1_reward": rewards[top_index],
        "oracle_reward": oracle_reward,
        "oracle_probability": sum(
            probability
            for probability, reward in zip(probabilities, rewards)
            if abs(float(reward) - float(oracle_reward)) < 1e-9
        ),
        "strict_probability": sum(
            probability
            for probability, item in zip(probabilities, support.feedback)
            if item.strict_success
        ),
        "property_all_probability": sum(
            probability
            for probability, item in zip(probabilities, support.feedback)
            if item.property_all_success
        ),
        "similarity_probability": sum(
            probability
            for probability, item in zip(probabilities, support.feedback)
            if item.source_similarity_success
        ),
        "top1_strict": support.feedback[top_index].strict_success,
        "top1_property_all": support.feedback[top_index].property_all_success,
        "top1_similarity": support.feedback[top_index].source_similarity_success,
        "support_strict_any": any(item.strict_success for item in support.feedback),
        "support_property_all_any": any(item.property_all_success for item in support.feedback),
        "support_similarity_any": any(item.source_similarity_success for item in support.feedback),
    }


def aggregate_action_value_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        return {"conditions": 0}

    def average(key: str) -> float:
        return mean(float(item[key]) for item in rows)

    return {
        "conditions": len(rows),
        "mean_support_size": average("support_size"),
        "mean_expected_reward": average("expected_reward"),
        "mean_top1_reward": average("top1_reward"),
        "mean_oracle_reward": average("oracle_reward"),
        "mean_oracle_probability": average("oracle_probability"),
        "mean_strict_probability": average("strict_probability"),
        "mean_property_all_probability": average("property_all_probability"),
        "mean_similarity_probability": average("similarity_probability"),
        "top1_strict_rate": average("top1_strict"),
        "top1_property_all_rate": average("top1_property_all"),
        "top1_similarity_rate": average("top1_similarity"),
        "support_strict_any_rate": average("support_strict_any"),
        "support_property_all_any_rate": average("support_property_all_any"),
        "support_similarity_any_rate": average("support_similarity_any"),
    }


def evaluate_action_values(
    model: object,
    tokenizer: object,
    records: Sequence[Mapping[str, object]],
    *,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    for record in records:
        ir = policy.constraint_ir(record)
        source = str(ir.get("source_smiles", "") or "").strip()
        support = evaluate_action_support(
            model,
            tokenizer,
            record,
            current_smiles=source,
            original_source_smiles=source,
            history=[],
            step_index=0,
            args=args,
        )
        if support is not None:
            rows.append(action_value_record(support, temperature=float(args.temperature)))
    by_origin = {
        origin: aggregate_action_value_rows(
            [item for item in rows if str(item.get("origin", "")) == origin]
        )
        for origin in requested_origins(args.rollout_origins)
    }
    return {"all": aggregate_action_value_rows(rows), "by_origin": by_origin}, rows


def select_retention_records(
    rows: Sequence[Mapping[str, object]],
    *,
    max_per_origin: int,
    seed: int,
) -> list[Mapping[str, object]]:
    grouped: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for index, row in enumerate(rows):
        origin = str(row.get("origin", "") or "").strip()
        if origin not in {"denovo", "table1", "mumo"}:
            continue
        identity = str(row.get("example_id", "") or f"{origin}:{index}")
        grouped[origin].setdefault(identity, row)
    missing = sorted({"denovo", "table1", "mumo"} - set(grouped))
    if missing:
        raise ValueError(f"Retention split is missing origins: {missing}")
    selected = []
    for offset, origin in enumerate(("denovo", "table1", "mumo")):
        candidates = list(grouped[origin].values())
        random.Random(int(seed) + offset).shuffle(candidates)
        selected.extend(candidates[: max(1, int(max_per_origin))])
    return selected


def evaluate_canonical_action_retention(
    model: object,
    tokenizer: object,
    records: Sequence[Mapping[str, object]],
    *,
    max_length: int,
) -> dict[str, object]:
    rows = []
    model.eval()
    for record in records:
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        assistant = messages[-1]
        if not isinstance(assistant, Mapping):
            continue
        try:
            payload = json.loads(str(assistant.get("content", "") or ""))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        score = inference_action_scores(
            model,
            tokenizer,
            messages[:-1],
            [payload],
            max_length=int(max_length),
            batch_size=1,
        )[0]
        rows.append({"origin": str(record.get("origin", "") or "unknown"), "score": score})

    def summarize(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "rows": len(items),
            "mean_canonical_action_log_probability": (
                mean(float(item["score"]) for item in items) if items else None
            ),
        }

    return {
        "all": summarize(rows),
        "by_origin": {
            origin: summarize([item for item in rows if item["origin"] == origin])
            for origin in ("denovo", "table1", "mumo")
        },
    }


def evaluate_records(
    model: object,
    tokenizer: object,
    records: Sequence[Mapping[str, object]],
    *,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, object], list[PolicyTrajectory]]:
    trajectories = []
    for index, record in enumerate(records):
        trajectories.extend(
            rollout_group(
                model,
                tokenizer,
                record,
                args=args,
                rollout_count=int(args.validation_rollouts),
                seed=int(seed) + 1009 * index,
            )
        )
    by_origin = {}
    for origin in requested_origins(args.rollout_origins):
        by_origin[origin] = aggregate_metrics([item for item in trajectories if item.origin == origin])
    return {"all": aggregate_metrics(trajectories), "by_origin": by_origin}, trajectories


def gate_metrics(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    baseline_action_values: Mapping[str, object] | None = None,
    candidate_action_values: Mapping[str, object] | None = None,
    baseline_retention: Mapping[str, object] | None = None,
    candidate_retention: Mapping[str, object] | None = None,
    retention_max_regression: float = 0.05,
) -> dict[str, object]:
    before = baseline["all"]
    after = candidate["all"]
    assert isinstance(before, Mapping) and isinstance(after, Mapping)
    reward_gain = float(after["mean_best_reward"]) - float(before["mean_best_reward"])
    strict_gain = float(after["strict_any_rate"]) - float(before["strict_any_rate"])
    property_gain = float(after["property_all_any_rate"]) - float(before["property_all_any_rate"])
    similarity_gain = float(after["source_similarity_any_rate"]) - float(before["source_similarity_any_rate"])
    action_expected_reward_gain = 0.0
    action_top1_reward_gain = 0.0
    action_property_probability_gain = 0.0
    action_similarity_probability_gain = 0.0
    if baseline_action_values is not None and candidate_action_values is not None:
        baseline_action_all = baseline_action_values["all"]
        candidate_action_all = candidate_action_values["all"]
        assert isinstance(baseline_action_all, Mapping) and isinstance(candidate_action_all, Mapping)
        action_expected_reward_gain = float(candidate_action_all["mean_expected_reward"]) - float(
            baseline_action_all["mean_expected_reward"]
        )
        action_top1_reward_gain = float(candidate_action_all["mean_top1_reward"]) - float(
            baseline_action_all["mean_top1_reward"]
        )
        action_property_probability_gain = float(
            candidate_action_all["mean_property_all_probability"]
        ) - float(baseline_action_all["mean_property_all_probability"])
        action_similarity_probability_gain = float(
            candidate_action_all["mean_similarity_probability"]
        ) - float(baseline_action_all["mean_similarity_probability"])

    retention_gains: dict[str, float] = {}
    if baseline_retention is not None and candidate_retention is not None:
        baseline_by_origin = baseline_retention.get("by_origin", {})
        candidate_by_origin = candidate_retention.get("by_origin", {})
        if isinstance(baseline_by_origin, Mapping) and isinstance(candidate_by_origin, Mapping):
            for origin in ("denovo", "table1", "mumo"):
                before = baseline_by_origin.get(origin, {})
                after_origin = candidate_by_origin.get(origin, {})
                if not isinstance(before, Mapping) or not isinstance(after_origin, Mapping):
                    continue
                before_value = before.get("mean_canonical_action_log_probability")
                after_value = after_origin.get("mean_canonical_action_log_probability")
                if before_value is not None and after_value is not None:
                    retention_gains[origin] = float(after_value) - float(before_value)

    primary_signal = (
        reward_gain >= 0.03
        or strict_gain > 0.0
        or property_gain > 0.0
        or action_expected_reward_gain >= 0.01
        or action_top1_reward_gain > 0.0
    )
    safe = (
        property_gain >= -0.02
        and similarity_gain >= -0.02
        and action_property_probability_gain >= -0.02
        and action_similarity_probability_gain >= -0.02
        and all(
            gain >= -float(retention_max_regression) for gain in retention_gains.values()
        )
    )
    return {
        "decision": "advance" if primary_signal and safe else "stop",
        "mean_best_reward_gain": reward_gain,
        "strict_any_rate_gain": strict_gain,
        "property_all_any_rate_gain": property_gain,
        "source_similarity_any_rate_gain": similarity_gain,
        "action_expected_reward_gain": action_expected_reward_gain,
        "action_top1_reward_gain": action_top1_reward_gain,
        "action_property_all_probability_gain": action_property_probability_gain,
        "action_similarity_probability_gain": action_similarity_probability_gain,
        "canonical_action_retention_gain_by_origin": retention_gains,
        "requirements": {
            "primary_signal": (
                "rollout reward +0.03, positive strict/property gain, exact expected reward +0.01, "
                "or positive exact top-1 reward gain"
            ),
            "max_property_regression": 0.02,
            "max_similarity_regression": 0.02,
            "max_canonical_action_log_probability_regression": float(retention_max_regression),
        },
    }


def write_trajectories(path: Path, trajectories: Sequence[PolicyTrajectory]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory_record(trajectory), sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(f"Missing common-LLM tool-policy dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("Common-LLM tool-policy training requires CUDA")
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    origins = requested_origins(args.rollout_origins)
    train_rows_all = policy.read_jsonl(args.train_jsonl)
    validation_rows_all = policy.read_jsonl(args.validation_jsonl)
    train_rows = unique_records(
        train_rows_all,
        origins=origins,
        max_per_origin=int(args.max_train_per_origin),
        seed=int(args.seed),
    )
    validation_rows = unique_records(
        validation_rows_all,
        origins=origins,
        max_per_origin=int(args.max_validation_per_origin),
        seed=int(args.seed) + 1,
    )
    retention_rows = select_retention_records(
        validation_rows_all,
        max_per_origin=int(args.retention_max_per_origin),
        seed=int(args.seed) + 2,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.input_adapter_dir, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model = model.cuda()

    anchors = anchor_examples(
        train_rows_all,
        tokenizer,
        max_per_origin=int(args.anchor_max_per_origin),
        max_length=int(args.max_length),
        seed=int(args.seed),
    )
    anchor_cursors = {origin: 0 for origin in anchors}

    baseline_action_values, baseline_action_value_rows = evaluate_action_values(
        model,
        tokenizer,
        validation_rows,
        args=args,
    )
    write_jsonl(
        args.output_dir / "baseline_action_values.jsonl",
        baseline_action_value_rows,
    )
    baseline_retention = evaluate_canonical_action_retention(
        model,
        tokenizer,
        retention_rows,
        max_length=int(args.max_length),
    )
    baseline_metrics, baseline_trajectories = evaluate_records(
        model,
        tokenizer,
        validation_rows,
        args=args,
        seed=int(args.seed) + 100_000,
    )
    write_trajectories(args.output_dir / "baseline_validation_trajectories.jsonl", baseline_trajectories)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(args.learning_rate), weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    history = []
    condition_step = 0
    update_step = 0
    all_train_trajectories: list[PolicyTrajectory] = []
    for epoch in range(1, int(args.epochs) + 1):
        shuffled = list(train_rows)
        random.Random(int(args.seed) + epoch).shuffle(shuffled)
        for record_index, record in enumerate(shuffled):
            model.eval()
            seed = int(args.seed) + epoch * 100_000 + record_index * 1009
            supports: list[PolicyStateSupport] = []
            if args.policy_update == "exact_action_value":
                supports, trajectories = rollout_exact_action_supports(
                    model,
                    tokenizer,
                    record,
                    args=args,
                    path_count=int(args.paths_per_condition),
                    seed=seed,
                )
            else:
                trajectories = rollout_group(
                    model,
                    tokenizer,
                    record,
                    args=args,
                    rollout_count=int(args.rollouts_per_condition),
                    seed=seed,
                )
            all_train_trajectories.extend(trajectories)
            rewards = [item.feedback.reward for item in trajectories]
            model.train()
            exact_diagnostics: dict[str, float] = {}
            if args.policy_update == "exact_action_value":
                policy_loss, decision_count, exact_diagnostics = exact_action_value_backward(
                    model,
                    tokenizer,
                    supports,
                    args=args,
                )
            else:
                advantages = policy.group_relative_advantages(rewards)
                policy_loss, decision_count = policy_gradient_backward(
                    model,
                    tokenizer,
                    trajectories,
                    advantages,
                    args=args,
                )
            anchor_loss = anchor_sft_backward(
                model,
                anchors,
                anchor_cursors,
                tokenizer=tokenizer,
                weight=float(args.anchor_sft_weight),
                gradient_accumulation=int(args.gradient_accumulation),
            )
            condition_step += 1
            do_update = (
                condition_step % int(args.gradient_accumulation) == 0
                or record_index + 1 == len(shuffled)
            )
            if do_update:
                torch.nn.utils.clip_grad_norm_(parameters, float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                nonfinite = common_train.adapter_nonfinite_count(model)
                if nonfinite:
                    raise FloatingPointError(
                        f"Detected {nonfinite} non-finite tool-policy adapter parameters at update {update_step}"
                    )
            row = {
                "epoch": epoch,
                "condition_step": condition_step,
                "origin": str(record.get("origin", "") or ""),
                "example_id": str(record.get("example_id", "") or ""),
                "mean_reward": mean(rewards),
                "best_reward": max(rewards),
                "reward_std": (
                    sum((value - mean(rewards)) ** 2 for value in rewards) / len(rewards)
                ) ** 0.5,
                "strict_any": any(item.feedback.strict_success for item in trajectories),
                "property_all_any": any(item.feedback.property_all_success for item in trajectories),
                "policy_loss": policy_loss,
                "anchor_sft_loss": anchor_loss,
                "decisions": decision_count,
                "updates": update_step,
                **exact_diagnostics,
            }
            history.append(row)
            if condition_step <= 2 or condition_step % int(args.logging_steps) == 0:
                print(json.dumps(row, sort_keys=True), flush=True)

    write_trajectories(args.output_dir / "train_trajectories.jsonl", all_train_trajectories)
    nonfinite = common_train.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"Refusing to save tool-policy adapter with {nonfinite} non-finite parameters")
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    candidate_action_values, candidate_action_value_rows = evaluate_action_values(
        model,
        tokenizer,
        validation_rows,
        args=args,
    )
    write_jsonl(
        args.output_dir / "candidate_action_values.jsonl",
        candidate_action_value_rows,
    )
    candidate_retention = evaluate_canonical_action_retention(
        model,
        tokenizer,
        retention_rows,
        max_length=int(args.max_length),
    )
    candidate_metrics, candidate_trajectories = evaluate_records(
        model,
        tokenizer,
        validation_rows,
        args=args,
        seed=int(args.seed) + 100_000,
    )
    write_trajectories(args.output_dir / "candidate_validation_trajectories.jsonl", candidate_trajectories)
    gate = gate_metrics(
        baseline_metrics,
        candidate_metrics,
        baseline_action_values=baseline_action_values,
        candidate_action_values=candidate_action_values,
        baseline_retention=baseline_retention,
        candidate_retention=candidate_retention,
        retention_max_regression=float(args.retention_max_regression),
    )
    summary = {
        "protocol": policy.POLICY_PROTOCOL,
        "method": (
            "typed_tool_exact_action_value_policy_iteration"
            if args.policy_update == "exact_action_value"
            else "on_policy_typed_tool_grpo"
        ),
        "gradient_estimator": (
            "exact_categorical_action_value_with_serial_autoregressive_backward"
            if args.policy_update == "exact_action_value"
            else "two_pass_executed_action_autoregressive_logprob"
        ),
        "action_support": "property_agnostic_universal_graph_edit_dsl_plus_rdkit",
        "property_evaluator": "rdkit_plus_pinned_tdc_plus_persistent_official_admet_ai",
        "complete_feedback_required": True,
        "target_or_candidate_pool_used_for_rollout": False,
        "base_model": args.base_model,
        "input_adapter_dir": str(args.input_adapter_dir),
        "adapter_dir": str(adapter_dir),
        "train_jsonl": str(args.train_jsonl),
        "validation_jsonl": str(args.validation_jsonl),
        "train_conditions": len(train_rows),
        "validation_conditions": len(validation_rows),
        "train_by_origin": dict(sorted(Counter(str(row.get("origin")) for row in train_rows).items())),
        "validation_by_origin": dict(
            sorted(Counter(str(row.get("origin")) for row in validation_rows).items())
        ),
        "rollouts_per_condition": int(args.rollouts_per_condition),
        "paths_per_condition": int(args.paths_per_condition),
        "validation_rollouts": int(args.validation_rollouts),
        "max_steps": int(args.max_steps),
        "max_actions": int(args.max_actions),
        "site_limit": int(args.site_limit),
        "anchor_origins": sorted(anchors),
        "anchor_counts": {key: len(value) for key, value in sorted(anchors.items())},
        "anchor_sft_weight": float(args.anchor_sft_weight),
        "epochs": int(args.epochs),
        "updates": update_step,
        "adapter_nonfinite_parameters": nonfinite,
        "baseline_validation": baseline_metrics,
        "candidate_validation": candidate_metrics,
        "baseline_action_values": baseline_action_values,
        "candidate_action_values": candidate_action_values,
        "baseline_canonical_action_retention": baseline_retention,
        "candidate_canonical_action_retention": candidate_retention,
        "retention_max_per_origin": int(args.retention_max_per_origin),
        "gate": gate,
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
