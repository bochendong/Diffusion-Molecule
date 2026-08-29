#!/usr/bin/env python3
"""Task- and mode-balanced group-relative RL for one shared MolProgram policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P26_DIR = SCRIPT_DIR.parent / "p26_decoupled_joint_rl"
P251_DIR = SCRIPT_DIR.parent / "p25_1_p23_mode_paired_grpo"
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
for path in (P26_DIR, P251_DIR, P25_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_decoupled_joint_rl as p26  # noqa: E402
import train_mode_paired_grpo as p251  # noqa: E402
import train_p23_joint_grpo as p25  # noqa: E402


DE_NOVO_BUCKETS = tuple(f"de_novo:{count}p" for count in range(2, 8))
EDIT_BUCKETS = tuple(f"edit:{task}" for task in sorted(p25.TARGET_EDIT_TASKS))


def stable_key(row: Mapping[str, object], seed: int) -> str:
    identity = row.get("example_id", row.get("sample_id", row.get("condition_id", "")))
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def balanced_bucket(row: Mapping[str, object]) -> str:
    mode = str(row.get("task_mode", ""))
    if mode == "de_novo":
        count = p25.property_count(row)
        return f"de_novo:{count}p" if 2 <= count <= 7 else ""
    if mode == "edit":
        task = str(row.get("task_key", ""))
        return f"edit:{task}" if task in p25.TARGET_EDIT_TASKS else ""
    return ""


def select_balanced_pairs(
    rows: Sequence[dict[str, object]], pairs: int, seed: int
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Return equal-mode pairs with equal exposure inside every task family."""
    if pairs <= 0 or pairs % 30:
        raise ValueError("pairs must be a positive multiple of 30")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        bucket = balanced_bucket(row)
        if bucket:
            grouped[bucket].append(row)
    per_denovo = pairs // len(DE_NOVO_BUCKETS)
    per_edit = pairs // len(EDIT_BUCKETS)
    selected_denovo: list[dict[str, object]] = []
    selected_edit: list[dict[str, object]] = []
    for bucket in DE_NOVO_BUCKETS:
        values = sorted(grouped[bucket], key=lambda row: stable_key(row, seed))
        if len(values) < per_denovo:
            raise ValueError(f"insufficient rows for {bucket}: {len(values)} < {per_denovo}")
        selected_denovo.extend(values[:per_denovo])
    for bucket in EDIT_BUCKETS:
        values = sorted(grouped[bucket], key=lambda row: stable_key(row, seed + 1))
        if len(values) < per_edit:
            raise ValueError(f"insufficient rows for {bucket}: {len(values)} < {per_edit}")
        selected_edit.extend(values[:per_edit])
    random.Random(seed + 2).shuffle(selected_denovo)
    random.Random(seed + 3).shuffle(selected_edit)
    if len(selected_denovo) != pairs or len(selected_edit) != pairs:
        raise AssertionError("balanced selector produced unequal mode exposure")
    return list(zip(selected_denovo, selected_edit))


def balanced_bisector_gradients(first, second):
    """Equal-norm bisector; a common descent direction unless exactly opposed."""
    import torch

    first_norm = torch.sqrt(sum(value.float().pow(2).sum() for value in first))
    second_norm = torch.sqrt(sum(value.float().pow(2).sum() for value in second))
    first_safe = first_norm.clamp_min(1e-12)
    second_safe = second_norm.clamp_min(1e-12)
    cosine = sum(
        (a.float() * b.float()).sum() for a, b in zip(first, second)
    ) / (first_safe * second_safe)
    scale = 0.5 * (first_norm + second_norm)
    merged = [
        0.5 * scale * (a / first_safe + b / second_safe)
        for a, b in zip(first, second)
    ]
    merged_norm = torch.sqrt(sum(value.float().pow(2).sum() for value in merged))
    dot_first = sum((value.float() * source.float()).sum() for value, source in zip(merged, first))
    dot_second = sum((value.float() * source.float()).sum() for value, source in zip(merged, second))
    return merged, {
        "gradient_cosine": float(cosine),
        "denovo_gradient_norm": float(first_norm),
        "edit_gradient_norm": float(second_norm),
        "merged_gradient_norm": float(merged_norm),
        "merged_dot_denovo": float(dot_first),
        "merged_dot_edit": float(dot_second),
        "common_descent": bool(float(dot_first) >= -1e-8 and float(dot_second) >= -1e-8),
    }


def group_record(row, rewards, details, advantage_record) -> dict[str, object]:
    record = p251.group_record(row, rewards, details)
    record["bucket"] = balanced_bucket(row)
    record["advantage"] = advantage_record
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reward-temperature", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1.5e-7)
    parser.add_argument("--denovo-anchor-weight", type=float, default=1.5)
    parser.add_argument("--edit-anchor-weight", type=float, default=1.5)
    parser.add_argument("--reference-kl-weight", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=30001)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("P30 requires BF16 CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = select_balanced_pairs(p25.read_jsonl(args.train_jsonl), args.pairs, args.seed)
    checkpoints = p251.complete_checkpoints(args.output_dir)
    start_step, history = 0, []
    policy_source = args.input_adapter
    resume_checkpoint = checkpoints[-1] if checkpoints else None
    if resume_checkpoint is not None:
        state = json.loads((resume_checkpoint / "state.json").read_text())
        start_step = int(state["next_step"])
        history = list(state["history"])
        policy_source = resume_checkpoint / "adapter"

    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
        loader_kind = "causal_lm"
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
        loader_kind = "image_text_to_text_text_only"
    else:
        raise TypeError(f"unsupported config: {type(config).__name__}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(
        base, policy_source, adapter_name="default", is_trainable=True
    ).cuda()
    model.load_adapter(args.input_adapter, adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate), weight_decay=0.0)
    if resume_checkpoint is not None:
        optimizer.load_state_dict(torch.load(resume_checkpoint / "optimizer.pt", map_location="cpu"))

    totals = Counter()
    live_log = args.output_dir / "training_history.live.jsonl"
    for step in range(start_step, len(schedule)):
        denovo_row, edit_row = schedule[step]
        sampled_groups = []
        for mode_index, row in enumerate((denovo_row, edit_row)):
            model.set_adapter("default")
            prompt_ids, candidates = p25.generate_group(
                model, tokenizer, list(row["messages"][:-1]), args.group_size,
                args.max_new_tokens, args.temperature, args.top_p,
                args.seed * 10000 + step * 2 + mode_index,
            )
            scored = [
                p26.reward_channels(row, candidate, temperature=args.reward_temperature)
                for candidate in candidates
            ]
            channel_rows = [item[0] for item in scored]
            details = [item[1] for item in scored]
            mode = str(row["task_mode"])
            advantages, advantage_record = p26.decoupled_advantages(
                channel_rows, p26.CHANNEL_WEIGHTS[mode]
            )
            scalar_rewards = [
                sum(
                    p26.CHANNEL_WEIGHTS[mode][name] * float(channels.get(name, 0.0))
                    for name in p26.CHANNEL_WEIGHTS[mode]
                )
                for channels in channel_rows
            ]
            sampled_groups.append((
                row, prompt_ids, candidates, scalar_rewards, details,
                advantages, advantage_record,
            ))

        mode_gradients = []
        mode_losses = []
        for mode_index, group in enumerate(sampled_groups):
            row, prompt_ids, candidates, _, _, advantages, _ = group
            optimizer.zero_grad(set_to_none=True)
            mode_losses.append(p26.backward_mode(
                model, tokenizer, row, prompt_ids, candidates, advantages,
                anchor_weight=(
                    args.denovo_anchor_weight if mode_index == 0 else args.edit_anchor_weight
                ),
                reference_kl_weight=args.reference_kl_weight,
            ))
            mode_gradients.append(p26.capture_gradients(trainable))

        merged, gradient_record = balanced_bisector_gradients(
            mode_gradients[0], mode_gradients[1]
        )
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(trainable, merged):
            parameter.grad = gradient
        unclipped_norm = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        optimizer.step()

        denovo_record = group_record(
            denovo_row, sampled_groups[0][3], sampled_groups[0][4], sampled_groups[0][6]
        )
        edit_record = group_record(
            edit_row, sampled_groups[1][3], sampled_groups[1][4], sampled_groups[1][6]
        )
        record = {
            "step": step,
            "de_novo": denovo_record,
            "edit": edit_record,
            "de_novo_loss": mode_losses[0],
            "edit_loss": mode_losses[1],
            **gradient_record,
            "unclipped_gradient_norm": float(unclipped_norm),
        }
        history.append(record)
        totals["paired_steps"] += 1
        totals["common_descent_steps"] += int(gradient_record["common_descent"])
        totals["gradient_conflicts"] += int(gradient_record["gradient_cosine"] < 0.0)
        totals["de_novo_zero_signal_groups"] += int(sampled_groups[0][6]["zero_signal"])
        totals["edit_zero_signal_groups"] += int(sampled_groups[1][6]["zero_signal"])
        with live_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps({"stage": "paired_step", **record}, sort_keys=True), flush=True)
        next_step = step + 1
        if next_step % args.checkpoint_every == 0 or next_step == len(schedule):
            checkpoint = p251.save_checkpoint(
                model, tokenizer, optimizer, args.output_dir, next_step, history
            )
            print(json.dumps({"stage": "checkpoint", "path": str(checkpoint)}), flush=True)

    model.set_adapter("default")
    nonfinite = sum(int((~torch.isfinite(parameter)).sum().item()) for parameter in trainable)
    if nonfinite:
        raise FloatingPointError(f"non-finite policy adapter parameters: {nonfinite}")
    adapter = args.output_dir / "adapter"
    p251.save_policy(model, adapter)
    tokenizer.save_pretrained(adapter)
    bucket_counts = Counter()
    for denovo_row, edit_row in schedule:
        bucket_counts[balanced_bucket(denovo_row)] += 1
        bucket_counts[balanced_bucket(edit_row)] += 1
    summary = {
        "protocol": "p30_balanced_shared_policy_group_relative_rl_v1",
        "loader_kind": loader_kind,
        "base_model": args.base_model,
        "input_adapter": str(args.input_adapter),
        "output_adapter": str(adapter),
        "paired_steps": len(schedule),
        "group_size": args.group_size,
        "learning_rate": args.learning_rate,
        "reward_temperature": args.reward_temperature,
        "channel_weights": p26.CHANNEL_WEIGHTS,
        "advantage": "per_mode_per_channel_group_zscore_then_renormalize",
        "gradient_merge": "equal_norm_bisector",
        "denovo_anchor_weight": args.denovo_anchor_weight,
        "edit_anchor_weight": args.edit_anchor_weight,
        "reference_kl_weight": args.reference_kl_weight,
        "reward_target_smiles_access": False,
        "sft_anchor_uses_training_positive": True,
        "bucket_group_counts": dict(sorted(bucket_counts.items())),
        "adapter_nonfinite_parameters": nonfinite,
        "totals": dict(sorted(totals.items())),
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
