#!/usr/bin/env python3
"""Evaluate only editing rows from the frozen P25.1 final gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
if str(P25_DIR) not in sys.path:
    sys.path.insert(0, str(P25_DIR))
import train_p23_joint_grpo as p25  # noqa: E402


def mean(values) -> float:
    values = list(values)
    return sum(float(value) for value in values) / max(len(values), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31151)
    args = parser.parse_args()

    import peft
    import torch
    import transformers

    rows = [row for row in p25.read_jsonl(args.gate_jsonl) if row["task_mode"] == "edit"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    loader = (
        transformers.AutoModelForCausalLM
        if type(config) in transformers.AutoModelForCausalLM._model_mapping
        else transformers.AutoModelForImageTextToText
    )
    base = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True

    records = []
    for repeat in range(args.repeats):
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            offset = encoded["input_ids"].shape[1]
            torch.manual_seed(args.seed + repeat * 100000 + start)
            with torch.no_grad():
                sampled = model.generate(
                    **encoded, max_new_tokens=128, do_sample=True,
                    temperature=0.8, top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for row, ids in zip(batch, sampled):
                raw = tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
                reward, details = p25.reward_response(row, raw)
                records.append({
                    "condition_id": row["condition_id"],
                    "bucket": p25.target_bucket(row),
                    "repeat": repeat,
                    "raw": raw,
                    "reward": reward,
                    **details,
                })
            print(
                f"[p31.1-edit:{args.method}] repeat={repeat + 1}/{args.repeats} "
                f"rows={min(start + args.batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )

    grouped = defaultdict(list)
    for record in records:
        grouped[record["bucket"]].append(record)
    buckets = {}
    for task in sorted(p25.TARGET_EDIT_TASKS):
        key = f"edit:{task}"
        items = grouped[key]
        if not items:
            raise ValueError(f"missing frozen editing bucket {key}")
        buckets[key] = {
            "candidates": len(items),
            "valid_rate": mean(item["valid"] for item in items),
            "strict_rate": mean(item["strict"] for item in items),
            "relaxed_rate": mean(item["relaxed"] for item in items),
        }
    summary = {
        "protocol": "p31_1_frozen_edit_raw1_gate_v1",
        "method": args.method,
        "rows": len(rows),
        "repeats": args.repeats,
        "candidates": len(records),
        "property_reranking": False,
        "aggregate": {
            "edit_valid_macro": mean(value["valid_rate"] for value in buckets.values()),
            "edit_strict_065_macro": mean(value["strict_rate"] for value in buckets.values()),
            "edit_relaxed_015_macro": mean(value["relaxed_rate"] for value in buckets.values())
        },
        "buckets": buckets
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
