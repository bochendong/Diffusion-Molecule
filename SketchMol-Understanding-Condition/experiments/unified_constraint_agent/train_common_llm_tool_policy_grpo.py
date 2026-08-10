#!/usr/bin/env python3
"""Train the common LLM as a closed-loop typed molecular tool policy.

Unlike the earlier preference experiments, this pilot does not consume a
ranked molecular candidate pool or mine strict-positive pairs.  It samples
executable GraphEditDSL calls from the current policy, executes them, returns
constraint feedback, and applies a group-relative policy-gradient update to
the complete one/two-step trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
            [decision.prompt_messages, decision.candidate_payloads],
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


def differentiable_action_scores(
    model: object,
    tokenizer: object,
    prompt_messages: Sequence[Mapping[str, object]],
    payloads: Sequence[Mapping[str, object]],
    *,
    max_length: int,
    batch_size: int,
):
    import torch

    encoded = [
        constrained.encoded_action(tokenizer, prompt_messages, payload, max_length=int(max_length))
        for payload in payloads
    ]
    pad_id = int(tokenizer.pad_token_id)
    scores = []
    for start in range(0, len(encoded), max(1, int(batch_size))):
        items = encoded[start : start + max(1, int(batch_size))]
        width = max(len(item["input_ids"]) for item in items)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in items:
            padding = width - len(item["input_ids"])
            batch["input_ids"].append([*item["input_ids"], *([pad_id] * padding)])
            batch["attention_mask"].append([*item["attention_mask"], *([0] * padding)])
            batch["labels"].append([*item["labels"], *([-100] * padding)])
        tensors = {
            key: torch.tensor(value, dtype=torch.long, device="cuda")
            for key, value in batch.items()
        }
        scores.append(common_train.sequence_mean_log_probability(model, tensors))
    return torch.cat(scores)


def policy_gradient_backward(
    model: object,
    tokenizer: object,
    trajectories: Sequence[PolicyTrajectory],
    advantages: Sequence[float],
    *,
    args: argparse.Namespace,
) -> tuple[float, int]:
    import torch

    grouped: dict[str, dict[str, object]] = {}
    total_decisions = 0
    for trajectory, advantage in zip(trajectories, advantages):
        for decision in trajectory.decisions:
            signature = _decision_signature(decision)
            group = grouped.setdefault(
                signature,
                {
                    "decision": decision,
                    "weights": defaultdict(float),
                    "count": 0,
                },
            )
            group["weights"][int(decision.selected_index)] += float(advantage)  # type: ignore[index]
            group["count"] = int(group["count"]) + 1
            total_decisions += 1
    if not grouped:
        return 0.0, 0
    detached_loss = 0.0
    normalizer = max(total_decisions, 1) * max(int(args.gradient_accumulation), 1)
    for group in grouped.values():
        decision = group["decision"]
        assert isinstance(decision, PolicyDecision)
        scores = differentiable_action_scores(
            model,
            tokenizer,
            decision.prompt_messages,
            decision.candidate_payloads,
            max_length=int(args.max_length),
            batch_size=int(args.score_batch_size),
        )
        log_probabilities = torch.log_softmax(scores / max(float(args.temperature), 1e-6), dim=0)
        loss = torch.zeros((), dtype=log_probabilities.dtype, device=log_probabilities.device)
        for index, weight in group["weights"].items():  # type: ignore[union-attr]
            loss = loss - float(weight) * log_probabilities[int(index)]
        detached_loss += float(loss.detach())
        (loss / normalizer).backward()
    return detached_loss / max(total_decisions, 1), total_decisions


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


def gate_metrics(baseline: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, object]:
    before = baseline["all"]
    after = candidate["all"]
    assert isinstance(before, Mapping) and isinstance(after, Mapping)
    reward_gain = float(after["mean_best_reward"]) - float(before["mean_best_reward"])
    strict_gain = float(after["strict_any_rate"]) - float(before["strict_any_rate"])
    property_gain = float(after["property_all_any_rate"]) - float(before["property_all_any_rate"])
    similarity_gain = float(after["source_similarity_any_rate"]) - float(before["source_similarity_any_rate"])
    primary_signal = reward_gain >= 0.03 or strict_gain > 0.0 or property_gain > 0.0
    safe = property_gain >= -0.02 and similarity_gain >= -0.02
    return {
        "decision": "advance" if primary_signal and safe else "stop",
        "mean_best_reward_gain": reward_gain,
        "strict_any_rate_gain": strict_gain,
        "property_all_any_rate_gain": property_gain,
        "source_similarity_any_rate_gain": similarity_gain,
        "requirements": {
            "primary_signal": "reward +0.03 or positive strict/property gain",
            "max_property_regression": 0.02,
            "max_similarity_regression": 0.02,
        },
    }


def write_trajectories(path: Path, trajectories: Sequence[PolicyTrajectory]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory_record(trajectory), sort_keys=True) + "\n")


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
            trajectories = rollout_group(
                model,
                tokenizer,
                record,
                args=args,
                rollout_count=int(args.rollouts_per_condition),
                seed=int(args.seed) + epoch * 100_000 + record_index * 1009,
            )
            all_train_trajectories.extend(trajectories)
            rewards = [item.feedback.reward for item in trajectories]
            advantages = policy.group_relative_advantages(rewards)
            model.train()
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

    candidate_metrics, candidate_trajectories = evaluate_records(
        model,
        tokenizer,
        validation_rows,
        args=args,
        seed=int(args.seed) + 100_000,
    )
    write_trajectories(args.output_dir / "candidate_validation_trajectories.jsonl", candidate_trajectories)
    gate = gate_metrics(baseline_metrics, candidate_metrics)
    summary = {
        "protocol": policy.POLICY_PROTOCOL,
        "method": "on_policy_typed_tool_grpo",
        "action_support": "property_agnostic_universal_graph_edit_dsl_plus_rdkit",
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
