#!/usr/bin/env python3
"""Generate greedy plus sampled P24 candidates and score them target-blind."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P26_DIR = SCRIPT_DIR.parent / "p26_decoupled_joint_rl"
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
for path in (P26_DIR, P25_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import train_decoupled_joint_rl as p26  # noqa: E402
import train_p23_joint_grpo as p25  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def seed_for(seed: int, shard: int, start: int) -> int:
    material = hashlib.sha256(f"{seed}:{shard}:{start}".encode()).hexdigest()
    return int(material[:8], 16)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=31001)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    rows = read_jsonl(args.input_jsonl)
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    loader = (
        transformers.AutoModelForCausalLM
        if type(config) in transformers.AutoModelForCausalLM._model_mapping
        else transformers.AutoModelForImageTextToText
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    batch_size = max(1, args.batch_size)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    list(row["messages"][:-1]), tokenize=False, add_generation_prompt=True
                )
                for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            offset = encoded["input_ids"].shape[1]
            with torch.no_grad():
                greedy_ids = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                torch.manual_seed(seed_for(args.seed, args.shard, start))
                sampled_ids = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, top_p=args.top_p,
                    num_return_sequences=args.sample_count,
                    pad_token_id=tokenizer.pad_token_id,
                )
            greedy = [
                tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
                for ids in greedy_ids
            ]
            sampled = [
                tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
                for ids in sampled_ids
            ]
            for row_index, row in enumerate(batch):
                raw_candidates = [greedy[row_index], *sampled[
                    row_index * args.sample_count : (row_index + 1) * args.sample_count
                ]]
                mode = str(row["task_mode"])
                weights = p26.CHANNEL_WEIGHTS[mode]
                scored = [p26.reward_channels(row, raw) for raw in raw_candidates]
                sampled_channels = [item[0] for item in scored[1:]]
                sampled_advantages, advantage_record = p26.decoupled_advantages(
                    sampled_channels, weights
                )
                candidates = []
                for index, (raw, (channels, details)) in enumerate(zip(raw_candidates, scored)):
                    parsed = p25.protocol.parse_response(raw, mode)
                    candidates.append({
                        "kind": "greedy" if index == 0 else "sampled",
                        "sample_index": None if index == 0 else index - 1,
                        "raw": raw,
                        "smiles": str(parsed.get("smiles", "") or ""),
                        "channels": channels,
                        "scalar_reward": sum(
                            float(weights[name]) * float(channels.get(name, 0.0))
                            for name in weights
                        ),
                        "advantage": None if index == 0 else sampled_advantages[index - 1],
                        **details,
                    })
                record = {
                    "protocol": "p31_p24_reward_support_candidates_v1",
                    "example_id": row.get("example_id", row.get("sample_id", "")),
                    "bucket": row["_audit_bucket"],
                    "mode": mode,
                    "prompt_messages": list(row["messages"][:-1]),
                    "target_access": False,
                    "sampled_advantage": advantage_record,
                    "candidates": candidates,
                }
                output.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"[p31-shard-{args.shard}] {min(start + batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )

    summary = {
        "protocol": "p31_p24_reward_support_shard_v1",
        "shard": args.shard,
        "prompts": len(rows),
        "candidates_per_prompt": 1 + args.sample_count,
        "greedy": True,
        "sampling": {
            "sample_count": args.sample_count,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "target_access": False,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
