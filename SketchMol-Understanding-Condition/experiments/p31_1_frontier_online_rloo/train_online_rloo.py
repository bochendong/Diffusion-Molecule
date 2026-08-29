#!/usr/bin/env python3
"""Frontier-conditioned online RLOO for one MolProgram task-mode adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P30_DIR = SCRIPT_DIR.parent / "p30_balanced_shared_policy_rl"
P26_DIR = SCRIPT_DIR.parent / "p26_decoupled_joint_rl"
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
for path in (P30_DIR, P26_DIR, P25_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_balanced_shared_rl as p30  # noqa: E402
import train_decoupled_joint_rl as p26  # noqa: E402
import train_p23_joint_grpo as p25  # noqa: E402
from rloo_math import rloo_advantages, scalar_reward  # noqa: E402


MODE_BUCKETS = {
    "de_novo": p30.DE_NOVO_BUCKETS,
    "edit": p30.EDIT_BUCKETS,
}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_key(row: Mapping[str, object], seed: int) -> str:
    identity = row.get("example_id", row.get("sample_id", row.get("condition_id", "")))
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def grouped_rows(rows: Sequence[dict[str, object]], mode: str, seed: int):
    expected = MODE_BUCKETS[mode]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("task_mode", "")) != mode:
            continue
        bucket = p30.balanced_bucket(row)
        if bucket in expected:
            grouped[bucket].append(row)
    for bucket in expected:
        grouped[bucket].sort(key=lambda row: stable_key(row, seed))
        if not grouped[bucket]:
            raise ValueError(f"no P31.1 training rows for {bucket}")
    return grouped


def scheduled_row(grouped, mode: str, attempt: int):
    buckets = MODE_BUCKETS[mode]
    bucket = buckets[attempt % len(buckets)]
    cycle = attempt // len(buckets)
    rows = grouped[bucket]
    return bucket, rows[cycle % len(rows)]


def completion_mask(generated, prompt_length: int, eos_token_id: int | None):
    import torch

    mask = torch.zeros_like(generated, dtype=torch.long)
    for row_index in range(generated.shape[0]):
        end = generated.shape[1]
        if eos_token_id is not None:
            positions = (generated[row_index, prompt_length:] == int(eos_token_id)).nonzero()
            if positions.numel():
                end = prompt_length + int(positions[0].item()) + 1
        mask[row_index, prompt_length:end] = 1
    return mask


def sequence_logprob_sums(model, generated, attention_mask, token_mask):
    import torch

    logits = model(input_ids=generated, attention_mask=attention_mask).logits[:, :-1].float()
    targets = generated[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1)
    chosen = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    shifted_mask = token_mask[:, 1:].to(chosen.dtype)
    return (chosen * shifted_mask).sum(dim=-1), shifted_mask.sum(dim=-1).clamp_min(1.0)


def generate_group(model, tokenizer, messages, group_size: int, max_new_tokens: int, seed: int):
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = encoded["input_ids"].shape[1]
    torch.manual_seed(int(seed))
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            num_return_sequences=int(group_size),
            pad_token_id=tokenizer.pad_token_id,
        )
    model.config.use_cache = False
    token_mask = completion_mask(generated, prompt_length, tokenizer.eos_token_id)
    prompt_mask = encoded["attention_mask"].repeat_interleave(group_size, dim=0)
    completion_attention = token_mask[:, prompt_length:]
    attention_mask = torch.cat((prompt_mask, completion_attention), dim=1)
    raw = [
        tokenizer.decode(ids[prompt_length:], skip_special_tokens=True).strip()
        for ids in generated
    ]
    return generated, attention_mask, token_mask, raw


def checkpoint_paths(output_dir: Path) -> list[Path]:
    return sorted(
        path for path in output_dir.glob("checkpoint-*")
        if (path / "CHECKPOINT_COMPLETE").is_file()
    )


def save_checkpoint(model, tokenizer, optimizer, output_dir: Path, updates: int, state) -> Path:
    import torch

    checkpoint = output_dir / f"checkpoint-{updates:03d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.set_adapter("default")
    model.save_pretrained(checkpoint / "adapter", selected_adapters=["default"])
    tokenizer.save_pretrained(checkpoint / "adapter")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    (checkpoint / "CHECKPOINT_COMPLETE").touch()
    return checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("de_novo", "edit"), required=True)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-updates", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=1200)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-7)
    parser.add_argument("--reference-kl-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--checkpoint-updates", default="25,50,100")
    parser.add_argument("--seed", type=int, default=31101)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("P31.1 requires BF16 CUDA")
    checkpoints_at = {int(value) for value in args.checkpoint_updates.split(",") if value}
    if args.target_updates not in checkpoints_at:
        checkpoints_at.add(args.target_updates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = grouped_rows(read_jsonl(args.train_jsonl), args.mode, args.seed)

    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
    else:
        raise TypeError(f"unsupported config: {type(config).__name__}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    completed = checkpoint_paths(args.output_dir)
    resume = completed[-1] if completed else None
    policy_source = resume / "adapter" if resume else args.input_adapter
    state = {
        "protocol": "p31_1_frontier_conditioned_online_rloo_v1",
        "mode": args.mode,
        "attempted_groups": 0,
        "informative_updates": 0,
        "skipped_all_success": 0,
        "skipped_all_failure": 0,
        "skipped_nonfinite_groups": 0,
        "bucket_updates": {},
    }
    if resume:
        state.update(json.loads((resume / "state.json").read_text()))
    state.setdefault("skipped_nonfinite_groups", 0)

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
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    if resume:
        optimizer.load_state_dict(torch.load(resume / "optimizer.pt", map_location="cpu"))

    live_log = args.output_dir / "training_history.live.jsonl"
    bucket_updates = Counter({key: int(value) for key, value in state["bucket_updates"].items()})
    while (
        int(state["informative_updates"]) < args.target_updates
        and int(state["attempted_groups"]) < args.max_attempts
    ):
        attempt = int(state["attempted_groups"])
        bucket, row = scheduled_row(grouped, args.mode, attempt)
        model.set_adapter("default")
        generated, attention_mask, token_mask, raw = generate_group(
            model, tokenizer, list(row["messages"][:-1]), args.group_size,
            args.max_new_tokens, args.seed * 100000 + attempt,
        )
        scored = [p26.reward_channels(row, candidate) for candidate in raw]
        channels = [item[0] for item in scored]
        details = [item[1] for item in scored]
        rewards = [scalar_reward(c, d, args.mode) for c, d in scored]
        strict = [bool(item.get("strict")) for item in details]
        state["attempted_groups"] = attempt + 1
        informative = any(strict) and not all(strict)
        record = {
            "attempt": attempt,
            "mode": args.mode,
            "bucket": bucket,
            "example_id": row.get("example_id", row.get("sample_id", "")),
            "informative": informative,
            "strict_count": sum(strict),
            "valid_count": sum(bool(item.get("valid")) for item in details),
            "reward_min": min(rewards),
            "reward_mean": sum(rewards) / len(rewards),
            "reward_max": max(rewards),
        }
        if not informative:
            key = "skipped_all_success" if all(strict) else "skipped_all_failure"
            state[key] = int(state[key]) + 1
            with live_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({"stage": "skip", **record}, sort_keys=True), flush=True)
            continue

        model.set_adapter("default")
        with torch.no_grad():
            policy_sums, lengths = sequence_logprob_sums(
                model, generated, attention_mask, token_mask
            )
        model.set_adapter("reference")
        with torch.no_grad():
            reference_sums, _ = sequence_logprob_sums(
                model, generated, attention_mask, token_mask
            )
        per_token_log_ratio = (policy_sums - reference_sums) / lengths
        regularized_returns = [
            reward - args.reference_kl_weight * float(log_ratio)
            for reward, log_ratio in zip(rewards, per_token_log_ratio)
        ]
        advantages = rloo_advantages(regularized_returns)

        model.set_adapter("default")
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for index, advantage in enumerate(advantages):
            sequence_sum, _ = sequence_logprob_sums(
                model,
                generated[index : index + 1],
                attention_mask[index : index + 1],
                token_mask[index : index + 1],
            )
            loss = -float(advantage) * sequence_sum.mean() / len(advantages)
            loss.backward()
            loss_value += float(loss.detach())
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in trainable
        )
        if gradients_finite:
            unclipped = torch.sqrt(sum(
                parameter.grad.float().pow(2).sum()
                for parameter in trainable if parameter.grad is not None
            ))
        else:
            unclipped = torch.tensor(float("nan"), device=model.device)
        record.update({
            "prospective_update": int(state["informative_updates"]) + 1,
            "loss": loss_value,
            "gradient_norm": float(unclipped),
            "advantage_min": min(advantages),
            "advantage_max": max(advantages),
            "advantage_sum": sum(advantages),
            "sampled_kl_per_token_mean": float(per_token_log_ratio.mean()),
        })
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite P31.1 loss: {record}")
        if not gradients_finite or not math.isfinite(float(unclipped)):
            state["skipped_nonfinite_groups"] = int(state["skipped_nonfinite_groups"]) + 1
            optimizer.zero_grad(set_to_none=True)
            with live_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"stage": "numerical_skip", **record}, sort_keys=True) + "\n")
            print(json.dumps({"stage": "numerical_skip", **record}, sort_keys=True), flush=True)
            continue

        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        state["informative_updates"] = int(state["informative_updates"]) + 1
        bucket_updates[bucket] += 1
        state["bucket_updates"] = dict(sorted(bucket_updates.items()))
        record["update"] = int(state["informative_updates"])
        with live_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps({"stage": "update", **record}, sort_keys=True), flush=True)

        updates = int(state["informative_updates"])
        if updates in checkpoints_at:
            path = save_checkpoint(model, tokenizer, optimizer, args.output_dir, updates, state)
            print(json.dumps({"stage": "checkpoint", "path": str(path)}, sort_keys=True), flush=True)

    if int(state["informative_updates"]) < args.target_updates:
        raise RuntimeError(
            f"only {state['informative_updates']} informative updates after "
            f"{state['attempted_groups']} attempted groups"
        )
    final_checkpoint = args.output_dir / f"checkpoint-{args.target_updates:03d}" / "adapter"
    final_adapter = args.output_dir / "adapter"
    if final_adapter.exists():
        shutil.rmtree(final_adapter)
    shutil.copytree(final_checkpoint, final_adapter)
    (args.output_dir / "TRAIN_COMPLETE").touch()
    print(json.dumps({"stage": "complete", **state}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
