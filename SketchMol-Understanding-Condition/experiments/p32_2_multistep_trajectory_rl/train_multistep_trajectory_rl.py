#!/usr/bin/env python3
"""Train P32.2 with diverse two-step groups and terminal-return credit."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P321_DIR = SCRIPT_DIR.parent / "p32_1_verifier_routed_residual_rl"
UCA_DIR = SCRIPT_DIR.parent / "unified_constraint_agent"
for path in (P321_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import residual_protocol as protocol  # noqa: E402
import train_residual_graph_rl as p321_train  # noqa: E402
import train_common_llm_preference as common_train  # noqa: E402
import train_common_llm_tool_policy_grpo as common_rl  # noqa: E402
from trajectory_math import centered_advantages  # noqa: E402


PROTOCOL = "p32_2_multistep_terminal_return_rl_v1"


@dataclass(frozen=True)
class SelectedAction:
    prompt_messages: list[dict[str, str]]
    payload: dict[str, object]
    kind: str


@dataclass(frozen=True)
class Trajectory:
    actions: tuple[SelectedAction, ...]
    final_feedback: protocol.CandidateFeedback
    final_smiles: str
    terminal_return: float


def categorical_probabilities(scores: Sequence[float], temperature: float) -> list[float]:
    probabilities, _advantages, _weights = common_rl.exact_action_distribution(
        scores, [0.0] * len(scores), temperature=float(temperature)
    )
    return probabilities


def weighted_without_replacement(
    scores: Sequence[float], *, count: int, temperature: float, seed: int
) -> list[int]:
    probabilities = categorical_probabilities(scores, temperature)
    available = list(range(len(scores)))
    selected = []
    rng = random.Random(int(seed))
    while available and len(selected) < int(count):
        mass = sum(probabilities[index] for index in available)
        draw = rng.random() * max(mass, 1e-20)
        cumulative = 0.0
        chosen = available[-1]
        for index in available:
            cumulative += probabilities[index]
            if draw <= cumulative:
                chosen = index
                break
        selected.append(chosen)
        available.remove(chosen)
    return selected


def categorical_index(scores: Sequence[float], *, temperature: float, seed: int) -> int:
    probabilities = categorical_probabilities(scores, temperature)
    rng = random.Random(int(seed))
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw <= cumulative:
            return index
    return len(scores) - 1


def terminal_return(feedback: protocol.CandidateFeedback, actions: Sequence[SelectedAction]) -> float:
    if feedback.strict_success:
        return 1.0
    if feedback.relaxed_success:
        return 0.2
    changed = any(action.kind != "stop" for action in actions)
    if feedback.valid and changed:
        return 0.02
    return 0.0


def observation(feedback: protocol.CandidateFeedback) -> dict[str, object]:
    return {
        "reward": feedback.reward,
        "valid": feedback.valid,
        "strict_success": feedback.strict_success,
        "relaxed_success": feedback.relaxed_success,
        "details": feedback.details,
    }


def rollout_group(
    model: object,
    tokenizer: object,
    record: Mapping[str, object],
    *,
    group_size: int,
    max_actions: int,
    site_limit: int,
    max_length: int,
    score_batch_size: int,
    temperature: float,
    seed: int,
) -> list[Trajectory]:
    if protocol.hard_accept_direct(record):
        raise ValueError("P32.2 training received a strict direct proposal")
    current = protocol.initial_smiles(record)
    first = protocol.support_bundle(
        model,
        tokenizer,
        record,
        current_smiles=current,
        history=[],
        step_index=0,
        max_steps=2,
        max_actions=max_actions,
        site_limit=site_limit,
        max_length=max_length,
        score_batch_size=score_batch_size,
    )
    if first is None:
        return []
    first_indices = weighted_without_replacement(
        first.support.action_scores,
        count=min(int(group_size), len(first.actions)),
        temperature=temperature,
        seed=seed,
    )
    trajectories = []
    for rollout_index, first_index in enumerate(first_indices):
        first_action = first.actions[first_index]
        first_feedback = first.support.feedback[first_index]
        selected = [SelectedAction(
            first.support.prompt_messages,
            first.support.candidate_payloads[first_index],
            first_action.kind,
        )]
        final_smiles = first_action.next_smiles
        final_feedback = first_feedback
        if not first_action.terminal:
            history = [{
                "tool_call": first_action.payload,
                "result_smiles": final_smiles,
                "observation": observation(first_feedback),
            }]
            second = protocol.support_bundle(
                model,
                tokenizer,
                record,
                current_smiles=final_smiles,
                history=history,
                step_index=1,
                max_steps=2,
                max_actions=max_actions,
                site_limit=site_limit,
                max_length=max_length,
                score_batch_size=score_batch_size,
            )
            if second is not None:
                second_index = categorical_index(
                    second.support.action_scores,
                    temperature=temperature,
                    seed=int(seed) + 1009 * (rollout_index + 1),
                )
                second_action = second.actions[second_index]
                final_smiles = second_action.next_smiles
                final_feedback = second.support.feedback[second_index]
                selected.append(SelectedAction(
                    second.support.prompt_messages,
                    second.support.candidate_payloads[second_index],
                    second_action.kind,
                ))
        trajectories.append(Trajectory(
            tuple(selected),
            final_feedback,
            final_smiles,
            terminal_return(final_feedback, selected),
        ))
    return trajectories


def trajectory_group_backward(
    model: object,
    tokenizer: object,
    trajectories: Sequence[Trajectory],
    *,
    temperature: float,
    max_length: int,
) -> tuple[float, dict[str, object]]:
    advantages = centered_advantages([trajectory.terminal_return for trajectory in trajectories])
    detached_loss = 0.0
    action_count = 0
    for trajectory, advantage in zip(trajectories, advantages):
        if abs(float(advantage)) < 1e-12:
            continue
        for action in trajectory.actions:
            score = common_rl.differentiable_selected_action_score(
                model,
                tokenizer,
                action.prompt_messages,
                action.payload,
                max_length=int(max_length),
            )
            loss = -float(advantage) * score / max(float(temperature), 1e-6)
            detached_loss += float(loss.detach())
            (loss / max(len(trajectories), 1)).backward()
            action_count += 1
    returns = [trajectory.terminal_return for trajectory in trajectories]
    return detached_loss / max(len(trajectories), 1), {
        "trajectories": len(trajectories),
        "backward_actions": action_count,
        "return_mean": mean(returns) if returns else 0.0,
        "return_max": max(returns) if returns else 0.0,
        "strict_trajectories": sum(item.final_feedback.strict_success for item in trajectories),
        "relaxed_trajectories": sum(item.final_feedback.relaxed_success for item in trajectories),
        "first_stop_trajectories": sum(item.actions[0].kind == "stop" for item in trajectories),
        "advantage_nonzero": sum(abs(value) >= 1e-12 for value in advantages),
    }


def save_checkpoint(model, tokenizer, output_dir: Path, step: int, history) -> Path:
    checkpoint = output_dir / f"checkpoint-{step:03d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint / "adapter")
    tokenizer.save_pretrained(checkpoint / "adapter")
    (checkpoint / "state.json").write_text(json.dumps({
        "protocol": PROTOCOL,
        "paired_updates": step,
        "history": history,
    }, indent=2, sort_keys=True) + "\n")
    (checkpoint / "CHECKPOINT_COMPLETE").touch()
    return checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--site-limit", type=int, default=24)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", default="10,20,30")
    parser.add_argument("--seed", type=int, default=32201)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    rows = protocol.read_jsonl(args.train_jsonl)
    by_mode = {
        mode: p321_train.select_failed(rows, mode, args.pairs, args.seed + offset)
        for offset, mode in enumerate(("de_novo", "edit"))
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(
        base, args.input_adapter, is_trainable=True
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    checkpoints = {int(value) for value in args.checkpoint_steps.split(",") if value}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for pair_index, (denovo, edit) in enumerate(
        zip(by_mode["de_novo"], by_mode["edit"]), start=1
    ):
        records = {"de_novo": denovo, "edit": edit}
        groups = {}
        model.eval()
        for offset, mode in enumerate(("de_novo", "edit")):
            groups[mode] = rollout_group(
                model,
                tokenizer,
                records[mode],
                group_size=args.group_size,
                max_actions=args.max_actions,
                site_limit=args.site_limit,
                max_length=args.max_length,
                score_batch_size=args.score_batch_size,
                temperature=args.temperature,
                seed=args.seed + pair_index * 10007 + offset,
            )
            if len(groups[mode]) < 2:
                raise RuntimeError(f"insufficient trajectory group for {mode} pair {pair_index}")

        gradients = {}
        diagnostics = {}
        for mode in ("de_novo", "edit"):
            optimizer.zero_grad(set_to_none=True)
            model.train()
            loss, group_diagnostics = trajectory_group_backward(
                model,
                tokenizer,
                groups[mode],
                temperature=args.temperature,
                max_length=args.max_length,
            )
            gradients[mode] = common_rl.snapshot_parameter_gradients(parameters)
            diagnostics[mode] = {
                "example_id": records[mode]["example_id"],
                "loss": loss,
                **group_diagnostics,
            }

        optimizer.zero_grad(set_to_none=True)
        has_gradient = any(
            gradient is not None
            for mode in gradients.values()
            for gradient in mode
        )
        if has_gradient:
            merge = common_rl.assign_paired_pcgrad(
                parameters, gradients["de_novo"], gradients["edit"]
            )
            finite = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            )
            if not finite:
                raise FloatingPointError(f"non-finite P32.2 gradient at pair {pair_index}")
            terms = [
                parameter.grad.float().pow(2).sum()
                for parameter in parameters if parameter.grad is not None
            ]
            unclipped = torch.sqrt(sum(terms)) if terms else torch.zeros(())
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
        else:
            merge = {
                "pcgrad_dot_product": 0.0,
                "pcgrad_cosine": 0.0,
                "pcgrad_projected": False,
                "pcgrad_left_norm": 0.0,
                "pcgrad_right_norm": 0.0,
            }
            unclipped = torch.zeros(())
        optimizer.zero_grad(set_to_none=True)
        nonfinite = common_train.adapter_nonfinite_count(model)
        if nonfinite:
            raise FloatingPointError(f"P32.2 adapter has {nonfinite} non-finite values")
        record = {
            "paired_update": pair_index,
            "optimizer_updated": has_gradient,
            "by_mode": diagnostics,
            "unclipped_gradient_norm": float(unclipped),
            **merge,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if pair_index in checkpoints:
            save_checkpoint(model, tokenizer, args.output_dir, pair_index, history)

    final = args.output_dir / f"checkpoint-{args.pairs:03d}" / "adapter"
    if not final.is_dir():
        save_checkpoint(model, tokenizer, args.output_dir, args.pairs, history)
    adapter = args.output_dir / "adapter"
    if adapter.exists():
        shutil.rmtree(adapter)
    shutil.copytree(final, adapter)
    summary = {
        "protocol": PROTOCOL,
        "paired_updates": args.pairs,
        "optimizer_updates": sum(bool(row["optimizer_updated"]) for row in history),
        "gradient_conflicts": sum(bool(row["pcgrad_projected"]) for row in history),
        "mean_gradient_cosine": mean(float(row["pcgrad_cosine"]) for row in history),
        "strict_trajectories": {
            mode: sum(int(row["by_mode"][mode]["strict_trajectories"]) for row in history)
            for mode in ("de_novo", "edit")
        },
        "adapter": str(adapter),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "TRAIN_COMPLETE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
