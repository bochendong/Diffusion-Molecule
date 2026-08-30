#!/usr/bin/env python3
"""Train P32.3 with strict-absorbing rollouts and editing support curriculum."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P321_DIR = SCRIPT_DIR.parent / "p32_1_verifier_routed_residual_rl"
P322_DIR = SCRIPT_DIR.parent / "p32_2_multistep_trajectory_rl"
UCA_DIR = SCRIPT_DIR.parent / "unified_constraint_agent"
for path in (P321_DIR, P322_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import residual_protocol as protocol  # noqa: E402
import train_residual_graph_rl as p321_train  # noqa: E402
import train_multistep_trajectory_rl as p322  # noqa: E402
import train_common_llm_preference as common_train  # noqa: E402
import train_common_llm_tool_policy_grpo as common_rl  # noqa: E402


PROTOCOL = "p32_3_strict_absorbing_exploration_rl_v1"


def select_supported_edit(
    rows: Sequence[Mapping[str, object]],
    support_rows: Sequence[Mapping[str, object]],
    count: int,
    seed: int,
    minimum_unique: int,
) -> tuple[list[Mapping[str, object]], int]:
    supported_ids = {
        str(row["example_id"])
        for row in support_rows
        if row.get("task_mode") == "edit" and bool(row.get("strict_rescue"))
    }
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("task_mode") == "edit"
            and str(row.get("example_id")) in supported_ids
            and not bool(row["direct_details"].get("strict"))
        ):
            grouped[str(row["bucket"])].append(row)
    unique = sum(len(bucket_rows) for bucket_rows in grouped.values())
    if unique < int(minimum_unique):
        raise ValueError(
            f"editing strict-support curriculum has {unique} unique rows; "
            f"requires {minimum_unique}"
        )
    for bucket, bucket_rows in grouped.items():
        bucket_rows.sort(key=lambda row: p321_train.stable_key(row, seed))
        random.Random(seed + len(bucket)).shuffle(bucket_rows)
    buckets = sorted(grouped)
    selected = []
    epoch = 0
    while len(selected) < int(count):
        order = list(buckets)
        random.Random(seed + epoch).shuffle(order)
        for bucket in order:
            for row in grouped[bucket]:
                selected.append(row)
                if len(selected) >= int(count):
                    break
            if len(selected) >= int(count):
                break
        epoch += 1
    return selected, unique


def rollout_group(
    model: object,
    tokenizer: object,
    record: Mapping[str, object],
    *,
    systematic_first_actions: bool,
    group_size: int,
    second_samples: int,
    max_actions: int,
    site_limit: int,
    max_length: int,
    score_batch_size: int,
    temperature: float,
    seed: int,
) -> list[p322.Trajectory]:
    """Sample terminal-return trajectories; strict feedback ends a path immediately."""
    if protocol.hard_accept_direct(record):
        raise ValueError("P32.3 training received a strict direct proposal")
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
    if systematic_first_actions:
        first_indices = list(range(len(first.actions)))
    else:
        first_indices = p322.weighted_without_replacement(
            first.support.action_scores,
            count=min(int(group_size), len(first.actions)),
            temperature=temperature,
            seed=seed,
        )
    trajectories = []
    for rollout_index, first_index in enumerate(first_indices):
        first_action = first.actions[first_index]
        first_feedback = first.support.feedback[first_index]
        selected_first = p322.SelectedAction(
            first.support.prompt_messages,
            first.support.candidate_payloads[first_index],
            first_action.kind,
        )
        # A verifier-confirmed strict molecule is an absorbing state. P32.2
        # incorrectly allowed the second graph edit to destroy this success.
        if first_action.terminal or first_feedback.strict_success:
            trajectories.append(p322.Trajectory(
                (selected_first,),
                first_feedback,
                first_action.next_smiles,
                p322.terminal_return(first_feedback, (selected_first,)),
            ))
            continue
        history = [{
            "tool_call": first_action.payload,
            "result_smiles": first_action.next_smiles,
            "observation": p322.observation(first_feedback),
        }]
        second = protocol.support_bundle(
            model,
            tokenizer,
            record,
            current_smiles=first_action.next_smiles,
            history=history,
            step_index=1,
            max_steps=2,
            max_actions=max_actions,
            site_limit=site_limit,
            max_length=max_length,
            score_batch_size=score_batch_size,
        )
        if second is None:
            trajectories.append(p322.Trajectory(
                (selected_first,),
                first_feedback,
                first_action.next_smiles,
                p322.terminal_return(first_feedback, (selected_first,)),
            ))
            continue
        second_indices = p322.weighted_without_replacement(
            second.support.action_scores,
            count=min(int(second_samples), len(second.actions)),
            temperature=temperature,
            seed=int(seed) + 1009 * (rollout_index + 1),
        )
        for second_index in second_indices:
            second_action = second.actions[second_index]
            second_feedback = second.support.feedback[second_index]
            selected_second = p322.SelectedAction(
                second.support.prompt_messages,
                second.support.candidate_payloads[second_index],
                second_action.kind,
            )
            actions = (selected_first, selected_second)
            trajectories.append(p322.Trajectory(
                actions,
                second_feedback,
                second_action.next_smiles,
                p322.terminal_return(second_feedback, actions),
            ))
    return trajectories


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
    parser.add_argument("--support-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--denovo-group-size", type=int, default=8)
    parser.add_argument("--edit-second-samples", type=int, default=4)
    parser.add_argument("--minimum-supported-edit", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--site-limit", type=int, default=24)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", default="5,10,20")
    parser.add_argument("--seed", type=int, default=32301)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    rows = protocol.read_jsonl(args.train_jsonl)
    support_rows = protocol.read_jsonl(args.support_jsonl)
    denovo = p321_train.select_failed(rows, "de_novo", args.pairs, args.seed)
    edit, supported_edit_unique = select_supported_edit(
        rows,
        support_rows,
        args.pairs,
        args.seed + 1,
        args.minimum_supported_edit,
    )
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

    for pair_index, (denovo_record, edit_record) in enumerate(zip(denovo, edit), start=1):
        records = {"de_novo": denovo_record, "edit": edit_record}
        model.eval()
        groups = {
            "de_novo": rollout_group(
                model, tokenizer, denovo_record,
                systematic_first_actions=False,
                group_size=args.denovo_group_size,
                second_samples=1,
                max_actions=args.max_actions,
                site_limit=args.site_limit,
                max_length=args.max_length,
                score_batch_size=args.score_batch_size,
                temperature=args.temperature,
                seed=args.seed + pair_index * 10007,
            ),
            "edit": rollout_group(
                model, tokenizer, edit_record,
                systematic_first_actions=True,
                group_size=args.denovo_group_size,
                second_samples=args.edit_second_samples,
                max_actions=args.max_actions,
                site_limit=args.site_limit,
                max_length=args.max_length,
                score_batch_size=args.score_batch_size,
                temperature=args.temperature,
                seed=args.seed + pair_index * 10007 + 1,
            ),
        }
        for mode, group in groups.items():
            if len(group) < 2:
                raise RuntimeError(f"insufficient P32.3 trajectory group for {mode} pair {pair_index}")

        gradients = {}
        diagnostics = {}
        for mode in ("de_novo", "edit"):
            optimizer.zero_grad(set_to_none=True)
            model.train()
            loss, group_diagnostics = p322.trajectory_group_backward(
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
            for mode_gradients in gradients.values()
            for gradient in mode_gradients
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
                raise FloatingPointError(f"non-finite P32.3 gradient at pair {pair_index}")
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
        if common_train.adapter_nonfinite_count(model):
            raise FloatingPointError(f"P32.3 adapter became non-finite at pair {pair_index}")
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
        "supported_edit_unique": supported_edit_unique,
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
