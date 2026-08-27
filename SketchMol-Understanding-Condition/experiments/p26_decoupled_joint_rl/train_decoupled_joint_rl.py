#!/usr/bin/env python3
"""Reward-decoupled, conflict-aware joint RL for the unified P23 policy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P251_DIR = SCRIPT_DIR.parent / "p25_1_p23_mode_paired_grpo"
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
for path in (P251_DIR, P25_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_mode_paired_grpo as p251  # noqa: E402
import train_p23_joint_grpo as p25  # noqa: E402


CHANNEL_WEIGHTS = {
    "de_novo": {
        "validity": 0.50,
        "canonical": 0.10,
        "property_mean": 0.75,
        "property_bottleneck": 1.00,
        "property_strict": 1.00,
    },
    "edit": {
        "validity": 0.50,
        "canonical": 0.10,
        "property_mean": 0.75,
        "property_bottleneck": 1.00,
        "property_strict": 1.00,
        "source_aligned": 0.50,
        "source_relaxed": 0.25,
        "source_strict": 0.50,
        "relaxed_success": 0.50,
        "strict_success": 1.00,
        "noncopy": 0.25,
    },
}


def reward_channels(
    row: Mapping[str, object], raw: str, *, temperature: float = 0.25
) -> tuple[dict[str, float], dict[str, object]]:
    """Return prompt-visible reward dimensions without scalarizing them."""
    mode = str(row["task_mode"])
    parsed = p25.protocol.parse_response(raw, mode)
    channels = {name: 0.0 for name in CHANNEL_WEIGHTS[mode]}
    details: dict[str, object] = {
        "valid": False,
        "canonical": False,
        "property_strict": False,
        "strict": False,
        "relaxed": False,
        "property_fraction": 0.0,
        "mean_satisfaction": 0.0,
        "bottleneck": 0.0,
        "source_similarity": None,
        "copy": False,
    }
    if not parsed.get("valid"):
        return channels, details

    channels["validity"] = 1.0
    channels["canonical"] = float(bool(parsed.get("canonical")))
    details["valid"] = True
    details["canonical"] = bool(parsed.get("canonical"))
    smiles = str(parsed["smiles"])
    score_row = p25.scorer_row(p25.prompt_payload(row), mode)
    scoring_mode = p25.unified.DE_NOVO_MODE if mode == "de_novo" else p25.unified.EDIT_MODE
    components = p25.unified.property_reward_components(
        score_row, smiles, mode=scoring_mode
    )
    mean_satisfaction = components.mean_satisfaction(float(temperature))
    softmin = components.softmin_margin(float(temperature))
    bottleneck = 0.5 * (math.tanh(float(softmin) / float(temperature)) + 1.0)
    property_strict = bool(components.all_success)
    channels["property_mean"] = float(mean_satisfaction)
    channels["property_bottleneck"] = float(bottleneck)
    channels["property_strict"] = float(property_strict)
    details.update({
        "property_strict": property_strict,
        "property_fraction": float(components.success_fraction),
        "mean_satisfaction": float(mean_satisfaction),
        "bottleneck": float(bottleneck),
    })

    strict = property_strict
    relaxed = property_strict
    if mode == "edit":
        source = str(score_row.get("source_smiles", "") or "")
        similarity = p25.unified.morgan_tanimoto(source, smiles)
        similarity = float(similarity) if math.isfinite(float(similarity)) else 0.0
        copy = bool(source and p25.unified.safe_canonical_smiles(source) == smiles)
        source_relaxed = similarity >= 0.15
        source_strict = similarity >= 0.65
        # Smooth signal centered at the paper's strict source threshold.
        source_aligned = 0.5 * (math.tanh((similarity - 0.65) / 0.15) + 1.0)
        relaxed = bool(property_strict and source_relaxed)
        strict = bool(property_strict and source_strict)
        channels.update({
            "source_aligned": float(source_aligned),
            "source_relaxed": float(source_relaxed),
            "source_strict": float(source_strict),
            "relaxed_success": float(relaxed),
            "strict_success": float(strict),
            "noncopy": float(not copy),
        })
        details["source_similarity"] = similarity
        details["copy"] = copy
    details["strict"] = strict
    details["relaxed"] = relaxed
    return channels, details


def zscores(values: Sequence[float], clip: float = 3.0) -> list[float]:
    center = sum(float(value) for value in values) / max(len(values), 1)
    variance = sum((float(value) - center) ** 2 for value in values) / max(len(values), 1)
    if variance < 1e-12:
        return [0.0 for _ in values]
    scale = variance**0.5
    return [max(-clip, min(clip, (float(value) - center) / scale)) for value in values]


def decoupled_advantages(
    channel_rows: Sequence[Mapping[str, float]],
    weights: Mapping[str, float],
) -> tuple[list[float], dict[str, object]]:
    """Normalize each reward dimension before aggregation, then renormalize."""
    combined = [0.0 for _ in channel_rows]
    active: list[str] = []
    for name, weight in weights.items():
        values = [float(row.get(name, 0.0)) for row in channel_rows]
        normalized = zscores(values)
        if any(abs(value) > 0.0 for value in normalized):
            active.append(name)
        for index, value in enumerate(normalized):
            combined[index] += float(weight) * value
    advantages = zscores(combined)
    return advantages, {
        "active_channels": active,
        "active_channel_count": len(active),
        "zero_signal": not any(abs(value) > 0.0 for value in advantages),
    }


def gradient_statistics(first, second) -> tuple[float, float, float, float]:
    import torch

    dot = sum((a.float() * b.float()).sum() for a, b in zip(first, second))
    norm_first_sq = sum(a.float().pow(2).sum() for a in first)
    norm_second_sq = sum(b.float().pow(2).sum() for b in second)
    cosine = dot / (norm_first_sq.sqrt() * norm_second_sq.sqrt()).clamp_min(1e-12)
    ratio = norm_first_sq.sqrt() / norm_second_sq.sqrt().clamp_min(1e-12)
    return float(cosine), float(ratio), float(dot), float(norm_first_sq + norm_second_sq)


def merge_gradients(first, second, *, surgery: str):
    """Symmetric two-task PCGrad; inputs already include equal mode weights."""
    import torch

    cosine, ratio, dot_value, _ = gradient_statistics(first, second)
    dot = sum((a.float() * b.float()).sum() for a, b in zip(first, second))
    norm_first_sq = sum(a.float().pow(2).sum() for a in first)
    norm_second_sq = sum(b.float().pow(2).sum() for b in second)
    conflict = bool(dot_value < 0.0)
    if surgery == "pcgrad" and conflict:
        first_out = [
            a - dot / norm_second_sq.clamp_min(1e-12) * b
            for a, b in zip(first, second)
        ]
        second_out = [
            b - dot / norm_first_sq.clamp_min(1e-12) * a
            for a, b in zip(first, second)
        ]
    else:
        first_out = list(first)
        second_out = list(second)
    merged = [a + b for a, b in zip(first_out, second_out)]
    merged_norm = torch.sqrt(sum(value.float().pow(2).sum() for value in merged))
    return merged, {
        "gradient_cosine": cosine,
        "gradient_norm_ratio_denovo_over_edit": ratio,
        "gradient_conflict": conflict,
        "gradient_surgery_applied": bool(surgery == "pcgrad" and conflict),
        "merged_gradient_norm": float(merged_norm),
    }


def capture_gradients(parameters):
    import torch

    return [
        parameter.grad.detach().clone()
        if parameter.grad is not None else torch.zeros_like(parameter)
        for parameter in parameters
    ]


def backward_mode(
    model,
    tokenizer,
    row,
    prompt_ids,
    candidates,
    advantages,
    *,
    anchor_weight: float,
    reference_kl_weight: float,
):
    model.eval()
    policy_loss_value = 0.0
    kl_values: list[float] = []
    for candidate, advantage in zip(candidates, advantages):
        mean_logprob, kl = p251.reference_kl_loss(
            model, tokenizer, prompt_ids, candidate
        )
        loss = (
            -0.5 * float(advantage) * mean_logprob / len(candidates)
            + 0.5 * float(reference_kl_weight) * kl / len(candidates)
        )
        loss.backward()
        policy_loss_value += float(loss.detach())
        kl_values.append(float(kl.detach()))
    model.set_adapter("default")
    model.train()
    anchor = p25.chosen_sft_loss(model, tokenizer, list(row["messages"]))
    (0.5 * float(anchor_weight) * anchor).backward()
    return {
        "policy_plus_kl_loss": policy_loss_value,
        "reference_kl_mean": sum(kl_values) / max(len(kl_values), 1),
        "sft_anchor_loss": float(anchor.detach()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reward-temperature", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--denovo-anchor-weight", type=float, default=1.0)
    parser.add_argument("--edit-anchor-weight", type=float, default=1.0)
    parser.add_argument("--reference-kl-weight", type=float, default=0.05)
    parser.add_argument("--gradient-surgery", choices=("none", "pcgrad"), default="pcgrad")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=26001)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("P26 requires BF16 CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = p251.select_mode_pairs(p25.read_jsonl(args.train_jsonl), args.pairs, args.seed)
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
        optimizer.load_state_dict(
            torch.load(resume_checkpoint / "optimizer.pt", map_location="cpu")
        )

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
                reward_channels(row, candidate, temperature=args.reward_temperature)
                for candidate in candidates
            ]
            channel_rows = [item[0] for item in scored]
            details = [item[1] for item in scored]
            mode = str(row["task_mode"])
            advantages, advantage_record = decoupled_advantages(
                channel_rows, CHANNEL_WEIGHTS[mode]
            )
            scalar_rewards = [
                sum(CHANNEL_WEIGHTS[mode][name] * float(channels.get(name, 0.0))
                    for name in CHANNEL_WEIGHTS[mode])
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
            mode_losses.append(backward_mode(
                model, tokenizer, row, prompt_ids, candidates, advantages,
                anchor_weight=(
                    args.denovo_anchor_weight if mode_index == 0 else args.edit_anchor_weight
                ),
                reference_kl_weight=args.reference_kl_weight,
            ))
            mode_gradients.append(capture_gradients(trainable))

        merged, gradient_record = merge_gradients(
            mode_gradients[0], mode_gradients[1], surgery=args.gradient_surgery
        )
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(trainable, merged):
            parameter.grad = gradient
        unclipped_norm = torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        optimizer.step()

        denovo_record = p251.group_record(
            denovo_row, sampled_groups[0][3], sampled_groups[0][4]
        )
        edit_record = p251.group_record(
            edit_row, sampled_groups[1][3], sampled_groups[1][4]
        )
        denovo_record["advantage"] = sampled_groups[0][6]
        edit_record["advantage"] = sampled_groups[1][6]
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
        totals["gradient_conflicts"] += int(gradient_record["gradient_conflict"])
        totals["gradient_surgeries"] += int(gradient_record["gradient_surgery_applied"])
        totals["de_novo_zero_signal_groups"] += int(
            sampled_groups[0][6]["zero_signal"]
        )
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
    summary = {
        "protocol": "p26_decoupled_conflict_aware_joint_rl_v1",
        "loader_kind": loader_kind,
        "base_model": args.base_model,
        "input_adapter": str(args.input_adapter),
        "output_adapter": str(adapter),
        "paired_steps": len(schedule),
        "group_size": args.group_size,
        "learning_rate": args.learning_rate,
        "reward_temperature": args.reward_temperature,
        "channel_weights": CHANNEL_WEIGHTS,
        "advantage": "per_mode_per_channel_group_zscore_then_renormalize",
        "gradient_surgery": args.gradient_surgery,
        "denovo_anchor_weight": args.denovo_anchor_weight,
        "edit_anchor_weight": args.edit_anchor_weight,
        "reference_kl_weight": args.reference_kl_weight,
        "reward_target_smiles_access": False,
        "sft_anchor_uses_training_positive": True,
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
