#!/usr/bin/env python3
"""Mode-paired P23 GRPO with an exact frozen-P23 adapter reference."""

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
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
for path in (P25_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_p23_joint_grpo as p25  # noqa: E402


def stable_key(row: Mapping[str, object], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row.get('example_id', '')}".encode()).hexdigest()


def select_mode_pairs(
    rows: Sequence[dict[str, object]], pairs: int, seed: int
) -> list[tuple[dict[str, object], dict[str, object]]]:
    if pairs <= 0 or pairs % 30:
        raise ValueError("pairs must be a positive multiple of 30")
    cycles = pairs // 30
    denovo: dict[str, list[dict[str, object]]] = defaultdict(list)
    edit: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        bucket = p25.target_bucket(row)
        if bucket.startswith("de_novo:"):
            denovo[bucket].append(row)
        elif bucket.startswith("edit:"):
            edit[bucket].append(row)
    selected_denovo: list[dict[str, object]] = []
    selected_edit: list[dict[str, object]] = []
    for bucket in ("de_novo:5p", "de_novo:6p", "de_novo:7p"):
        values = sorted(denovo[bucket], key=lambda row: stable_key(row, seed))
        needed = 10 * cycles
        if len(values) < needed:
            raise ValueError(f"insufficient rows for {bucket}: {len(values)} < {needed}")
        selected_denovo.extend(values[:needed])
    for bucket in sorted(f"edit:{task}" for task in p25.TARGET_EDIT_TASKS):
        values = sorted(edit[bucket], key=lambda row: stable_key(row, seed + 1))
        needed = 3 * cycles
        if len(values) < needed:
            raise ValueError(f"insufficient rows for {bucket}: {len(values)} < {needed}")
        selected_edit.extend(values[:needed])
    random.Random(seed + 2).shuffle(selected_denovo)
    random.Random(seed + 3).shuffle(selected_edit)
    if len(selected_denovo) != pairs or len(selected_edit) != pairs:
        raise AssertionError("mode-paired selector produced an imbalanced schedule")
    return list(zip(selected_denovo, selected_edit))


def completion_token_logprobs(model, tokenizer, prompt_ids, answer: str):
    import torch

    suffix = str(answer) + (tokenizer.eos_token or "")
    answer_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if answer_ids.numel() == 0:
        answer_ids = torch.tensor([[int(tokenizer.eos_token_id)]], dtype=torch.long)
    ids = torch.cat((prompt_ids.to(dtype=torch.long), answer_ids), dim=1).to(model.device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, :-1].float()
    targets = ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    positions = torch.arange(targets.shape[1], device=ids.device) + 1
    return token_logprobs[0, positions.ge(prompt_ids.shape[1])]


def reference_kl_loss(model, tokenizer, prompt_ids, candidate: str):
    import torch

    model.set_adapter("reference")
    with torch.no_grad():
        reference = completion_token_logprobs(model, tokenizer, prompt_ids, candidate).detach()
    model.set_adapter("default")
    policy = completion_token_logprobs(model, tokenizer, prompt_ids, candidate)
    log_ratio = policy - reference
    # Schulman's non-negative k3 estimator for KL(policy || reference).
    kl = (torch.exp(-log_ratio) + log_ratio - 1.0).mean()
    return policy.mean(), kl


def complete_checkpoints(output_dir: Path) -> list[Path]:
    return sorted(
        path for path in output_dir.glob("checkpoint-*")
        if (path / "CHECKPOINT_COMPLETE").is_file()
    )


def save_policy(model, path: Path) -> None:
    model.save_pretrained(path, selected_adapters=["default"])


def save_checkpoint(model, tokenizer, optimizer, output_dir: Path, next_step: int, history) -> Path:
    import torch

    checkpoint = output_dir / f"checkpoint-{next_step:03d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_policy(model, checkpoint / "adapter")
    tokenizer.save_pretrained(checkpoint / "adapter")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(json.dumps({
        "next_step": next_step, "history": history,
    }, indent=2, sort_keys=True) + "\n")
    (checkpoint / "CHECKPOINT_COMPLETE").touch()
    return checkpoint


def group_record(row, rewards, details) -> dict[str, object]:
    mean_reward = sum(rewards) / len(rewards)
    return {
        "example_id": row["example_id"],
        "bucket": p25.target_bucket(row),
        "mean_reward": mean_reward,
        "reward_std": (
            sum((value - mean_reward) ** 2 for value in rewards) / len(rewards)
        ) ** 0.5,
        "valid_fraction": sum(bool(item["valid"]) for item in details) / len(details),
        "property_strict_fraction": (
            sum(bool(item["property_strict"]) for item in details) / len(details)
        ),
        "strict_fraction": sum(bool(item["strict"]) for item in details) / len(details),
        "relaxed_fraction": sum(bool(item["relaxed"]) for item in details) / len(details),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--sft-anchor-weight", type=float, default=1.0)
    parser.add_argument("--reference-kl-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=25125)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("P25.1 requires BF16 CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = select_mode_pairs(p25.read_jsonl(args.train_jsonl), args.pairs, args.seed)
    checkpoints = complete_checkpoints(args.output_dir)
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
    optimizer = torch.optim.AdamW(
        trainable, lr=float(args.learning_rate), weight_decay=0.0
    )
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
            scored = [p25.reward_response(row, candidate) for candidate in candidates]
            rewards = [item[0] for item in scored]
            details = [item[1] for item in scored]
            sampled_groups.append((row, prompt_ids, candidates, rewards, details))

        optimizer.zero_grad(set_to_none=True)
        model.eval()
        policy_loss_value = 0.0
        kl_values = []
        for row, prompt_ids, candidates, rewards, _ in sampled_groups:
            advantages = p25.group_advantages(rewards)
            for candidate, advantage in zip(candidates, advantages):
                mean_logprob, kl = reference_kl_loss(
                    model, tokenizer, prompt_ids, candidate
                )
                loss = (
                    -0.5 * float(advantage) * mean_logprob / len(candidates)
                    + 0.5 * float(args.reference_kl_weight) * kl / len(candidates)
                )
                loss.backward()
                policy_loss_value += float(loss.detach())
                kl_values.append(float(kl.detach()))
        model.set_adapter("default")
        model.train()
        denovo_anchor = p25.chosen_sft_loss(model, tokenizer, list(denovo_row["messages"]))
        edit_anchor = p25.chosen_sft_loss(model, tokenizer, list(edit_row["messages"]))
        anchor = 0.5 * (denovo_anchor + edit_anchor)
        (float(args.sft_anchor_weight) * anchor).backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        optimizer.step()

        denovo_record = group_record(
            denovo_row, sampled_groups[0][3], sampled_groups[0][4]
        )
        edit_record = group_record(edit_row, sampled_groups[1][3], sampled_groups[1][4])
        record = {
            "step": step,
            "de_novo": denovo_record,
            "edit": edit_record,
            "policy_plus_kl_loss": policy_loss_value,
            "reference_kl_mean": sum(kl_values) / max(len(kl_values), 1),
            "sft_anchor_loss": float(anchor.detach()),
        }
        history.append(record)
        totals["paired_steps"] += 1
        totals["de_novo_groups"] += 1
        totals["edit_groups"] += 1
        with live_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps({"stage": "paired_step", **record}, sort_keys=True), flush=True)
        next_step = step + 1
        if next_step % args.checkpoint_every == 0 or next_step == len(schedule):
            checkpoint = save_checkpoint(
                model, tokenizer, optimizer, args.output_dir, next_step, history
            )
            print(json.dumps({"stage": "checkpoint", "path": str(checkpoint)}), flush=True)

    model.set_adapter("default")
    nonfinite = sum(int((~torch.isfinite(parameter)).sum().item()) for parameter in trainable)
    if nonfinite:
        raise FloatingPointError(f"non-finite policy adapter parameters: {nonfinite}")
    adapter = args.output_dir / "adapter"
    save_policy(model, adapter)
    tokenizer.save_pretrained(adapter)
    summary = {
        "protocol": "p25_1_p23_mode_paired_grpo_v1",
        "loader_kind": loader_kind,
        "base_model": args.base_model,
        "input_adapter": str(args.input_adapter),
        "reference_adapter": str(args.input_adapter),
        "output_adapter": str(adapter),
        "paired_steps": len(schedule),
        "de_novo_groups": len(schedule),
        "edit_groups": len(schedule),
        "group_size": args.group_size,
        "learning_rate": args.learning_rate,
        "sft_anchor_weight": args.sft_anchor_weight,
        "reference_kl_weight": args.reference_kl_weight,
        "reference_kl_estimator": "token_mean_schulman_k3",
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
