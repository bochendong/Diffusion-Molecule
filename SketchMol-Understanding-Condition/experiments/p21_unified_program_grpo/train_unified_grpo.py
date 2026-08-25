#!/usr/bin/env python3
"""Continue P18 with verifier-aligned, critic-free group-relative RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P17_DIR = SCRIPT_DIR.parent / "p17_copy_contrastive_unified_benchmark"
UNIFIED_DIR = SCRIPT_DIR.parent / "unified_smiles_generator"
for path in (P17_DIR, UNIFIED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import p17_protocol as protocol  # noqa: E402
import unified_smiles_generator as unified  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_payload(row: Mapping[str, object]) -> dict[str, object]:
    messages = list(row["messages"])
    for message in messages:
        if str(message.get("role")) == "user":
            payload = json.loads(str(message["content"]))
            if not isinstance(payload, dict):
                break
            return payload
    raise ValueError(f"missing structured user payload for {row.get('example_id')}")


def property_count(row: Mapping[str, object]) -> int:
    conditions = prompt_payload(row).get("conditions", [])
    return len(conditions) if isinstance(conditions, list) else 0


def stable_key(row: Mapping[str, object], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row.get('example_id', '')}".encode()).hexdigest()


def balanced_mode_rows(
    rows: Sequence[dict[str, object]], mode: str, limit: int, seed: int
) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row.get("task_mode")) == mode:
            grouped[property_count(row)].append(row)
    for values in grouped.values():
        values.sort(key=lambda item: stable_key(item, seed))
    selected: list[dict[str, object]] = []
    counts = sorted(grouped, reverse=True)
    while len(selected) < limit and counts:
        remaining = []
        for count in counts:
            if grouped[count] and len(selected) < limit:
                selected.append(grouped[count].pop(0))
            if grouped[count]:
                remaining.append(count)
        counts = remaining
    if len(selected) != limit:
        raise ValueError(f"insufficient {mode} rows: requested={limit} selected={len(selected)}")
    return selected


def select_training_rows(
    rows: Sequence[dict[str, object]], max_prompts: int, seed: int
) -> list[dict[str, object]]:
    if max_prompts < 2 or max_prompts % 2:
        raise ValueError("max_prompts must be an even integer >=2")
    per_mode = max_prompts // 2
    chosen = [
        *balanced_mode_rows(rows, "de_novo", per_mode, seed),
        *balanced_mode_rows(rows, "edit", per_mode, seed + 1),
    ]
    random.Random(seed).shuffle(chosen)
    return chosen


def scorer_row(payload: Mapping[str, object], mode: str) -> dict[str, str]:
    source = str(payload.get("source", "") or "")
    if source == "<EMPTY>":
        source = ""
    conditions = payload.get("conditions", [])
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("property program is missing conditions")
    result: dict[str, str] = {"source_smiles": source, "task_mode": mode}
    selected = []
    instruction = []
    for item in conditions:
        if not isinstance(item, Mapping):
            continue
        prop = str(item.get("property", "") or "")
        if prop not in protocol.PROPERTIES:
            continue
        goal = item.get("goal")
        selected.append(prop)
        result[f"{prop}_active"] = "True"
        if isinstance(goal, str) and goal in {"increase", "decrease", "preserve"}:
            result[f"{prop}_direction"] = goal
            instruction.append({"property": prop, "direction": goal})
        elif isinstance(goal, Mapping) and "around" in goal:
            result[f"target_{prop}"] = str(float(goal["around"]))
    if not selected:
        raise ValueError("property program has no supported properties")
    result["condition_properties"] = ",".join(selected)
    if instruction:
        result["instruction_tasks"] = json.dumps(instruction, separators=(",", ":"))
    return result


def reward_response(row: Mapping[str, object], raw: str) -> tuple[float, dict[str, object]]:
    mode = str(row["task_mode"])
    parsed = protocol.parse_response(raw, mode)
    if not parsed.get("valid"):
        return -1.0, {
            "valid": False,
            "canonical": False,
            "strict": False,
            "property_fraction": 0.0,
            "source_similarity": None,
        }
    smiles = str(parsed["smiles"])
    payload = prompt_payload(row)
    score_row = scorer_row(payload, mode)
    unified_mode = unified.DE_NOVO_MODE if mode == "de_novo" else unified.EDIT_MODE
    components = unified.property_reward_components(score_row, smiles, mode=unified_mode)
    mean_satisfaction = components.mean_satisfaction(0.25)
    softmin = components.softmin_margin(0.25)
    bottleneck = 0.5 * (math.tanh(float(softmin) / 0.25) + 1.0)
    reward = (
        0.5
        + 0.1 * float(bool(parsed.get("canonical")))
        + 0.75 * float(components.success_fraction)
        + 0.75 * float(mean_satisfaction)
        + 0.75 * float(bottleneck)
        + 1.0 * float(components.all_success)
    )
    source_similarity = None
    strict = bool(components.all_success)
    copy = False
    if mode == "edit":
        source = str(score_row.get("source_smiles", "") or "")
        similarity = unified.morgan_tanimoto(source, smiles)
        source_similarity = float(similarity) if math.isfinite(float(similarity)) else 0.0
        similarity_gate = source_similarity >= 0.65
        copy = bool(source and unified.safe_canonical_smiles(source) == smiles)
        strict = bool(components.all_success and similarity_gate)
        reward += 0.25 * min(source_similarity / 0.65, 1.0)
        reward += 0.50 * float(similarity_gate)
        reward += 1.00 * float(strict)
        reward -= 0.25 * float(copy)
    return float(reward), {
        "valid": True,
        "canonical": bool(parsed.get("canonical")),
        "strict": strict,
        "property_fraction": float(components.success_fraction),
        "mean_satisfaction": float(mean_satisfaction),
        "bottleneck": float(bottleneck),
        "source_similarity": source_similarity,
        "copy": copy,
    }


def group_advantages(rewards: Sequence[float], clip: float = 3.0) -> list[float]:
    center = sum(float(value) for value in rewards) / max(len(rewards), 1)
    variance = sum((float(value) - center) ** 2 for value in rewards) / max(len(rewards), 1)
    scale = max(variance**0.5, 1e-6)
    return [max(-clip, min(clip, (float(value) - center) / scale)) for value in rewards]


def completion_mean_logprob(model, tokenizer, prompt_ids, answer: str):
    import torch

    suffix = str(answer) + (tokenizer.eos_token or "")
    answer_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if answer_ids.numel() == 0:
        answer_ids = torch.tensor([[int(tokenizer.eos_token_id)]], dtype=torch.long)
    prompt_ids = prompt_ids.to(dtype=torch.long)
    ids = torch.cat((prompt_ids, answer_ids), dim=1).to(model.device)
    attention = torch.ones_like(ids)
    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1].float()
    targets = ids[:, 1:]
    token_logprob = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(targets.shape[1], device=ids.device) + 1
    mask = positions.ge(prompt_ids.shape[1]).to(token_logprob.dtype).unsqueeze(0)
    return (token_logprob * mask).sum() / mask.sum().clamp_min(1.0)


def chosen_sft_loss(model, tokenizer, messages: Sequence[Mapping[str, str]]):
    import torch

    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    answer = str(messages[-1]["content"]) + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat((prompt_ids, answer_ids), dim=1).to(model.device)
    labels = ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss.float()


def generate_group(model, tokenizer, messages, group_size: int, max_new_tokens: int, temperature: float, top_p: float, seed: int):
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    offset = encoded["input_ids"].shape[1]
    torch.manual_seed(int(seed))
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=int(group_size),
            pad_token_id=tokenizer.pad_token_id,
        )
    model.config.use_cache = False
    return encoded["input_ids"].detach().cpu(), [
        tokenizer.decode(ids[offset:], skip_special_tokens=True).strip() for ids in generated
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-prompts", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--sft-anchor-weight", type=float, default=0.20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2121)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("P21 requires BF16 CUDA")
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
        loader_kind = "causal_lm"
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
        loader_kind = "image_text_to_text_text_only"
    else:
        raise TypeError(f"unsupported config: {type(config).__name__}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.input_adapter, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate), weight_decay=0.0)

    selected = select_training_rows(read_jsonl(args.train_jsonl), args.max_prompts, args.seed)
    history = []
    totals = Counter()
    for index, row in enumerate(selected):
        prompt_messages = list(row["messages"][:-1])
        prompt_ids, candidates = generate_group(
            model, tokenizer, prompt_messages, args.group_size, args.max_new_tokens,
            args.temperature, args.top_p, args.seed * 1000 + index,
        )
        scored = [reward_response(row, candidate) for candidate in candidates]
        rewards = [item[0] for item in scored]
        details = [item[1] for item in scored]
        advantages = group_advantages(rewards)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        policy_loss_value = 0.0
        for candidate, advantage in zip(candidates, advantages):
            logprob = completion_mean_logprob(model, tokenizer, prompt_ids, candidate)
            loss = -float(advantage) * logprob / max(len(candidates), 1)
            loss.backward()
            policy_loss_value += float(loss.detach())
        anchor = chosen_sft_loss(model, tokenizer, list(row["messages"]))
        (float(args.sft_anchor_weight) * anchor).backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        optimizer.step()

        valid = sum(bool(item["valid"]) for item in details) / len(details)
        strict = sum(bool(item["strict"]) for item in details) / len(details)
        mode = str(row["task_mode"])
        totals[f"{mode}_groups"] += 1
        totals[f"{mode}_valid_candidates"] += sum(bool(item["valid"]) for item in details)
        totals[f"{mode}_strict_candidates"] += sum(bool(item["strict"]) for item in details)
        history.append({
            "index": index,
            "example_id": row["example_id"],
            "mode": mode,
            "property_count": property_count(row),
            "mean_reward": sum(rewards) / len(rewards),
            "reward_std": (sum((value - sum(rewards) / len(rewards)) ** 2 for value in rewards) / len(rewards)) ** 0.5,
            "valid_fraction": valid,
            "strict_fraction": strict,
            "policy_loss": policy_loss_value,
            "sft_anchor_loss": float(anchor.detach()),
        })
        print(json.dumps({"stage": "group", **history[-1]}, sort_keys=True), flush=True)

    nonfinite = sum(int((~torch.isfinite(parameter)).sum().item()) for parameter in trainable)
    if nonfinite:
        raise FloatingPointError(f"non-finite trainable adapter parameters: {nonfinite}")
    adapter = args.output_dir / "adapter"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    mode_counts = Counter(str(row["task_mode"]) for row in selected)
    property_counts = Counter(f"{row['task_mode']}:{property_count(row)}" for row in selected)
    summary = {
        "protocol": "p21_unified_program_grpo_v1",
        "loader_kind": loader_kind,
        "base_model": args.base_model,
        "input_adapter": str(args.input_adapter),
        "output_adapter": str(adapter),
        "prompts": len(selected),
        "mode_counts": dict(sorted(mode_counts.items())),
        "property_counts": dict(sorted(property_counts.items())),
        "group_size": args.group_size,
        "learning_rate": args.learning_rate,
        "sft_anchor_weight": args.sft_anchor_weight,
        "critic": False,
        "ppo_ratio_clipping": False,
        "reference_kl_weight": 0.0,
        "target_smiles_used_by_reward": False,
        "adapter_nonfinite_parameters": nonfinite,
        "totals": dict(sorted(totals.items())),
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
