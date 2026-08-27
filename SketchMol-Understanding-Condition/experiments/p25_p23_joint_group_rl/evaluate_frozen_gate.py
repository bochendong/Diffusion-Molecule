#!/usr/bin/env python3
"""Evaluate raw sampled policy behavior on the frozen P25 joint gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import train_p23_joint_grpo as rl


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
    parser.add_argument("--seed", type=int, default=25251)
    args = parser.parse_args()

    import peft
    import torch
    import transformers

    rows = rl.read_jsonl(args.gate_jsonl)
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

    records: list[dict[str, object]] = []
    batch_size = max(1, args.batch_size)
    for repeat in range(args.repeats):
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompt_text = [
                tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            encoded = tokenizer(prompt_text, return_tensors="pt", padding=True).to(model.device)
            offset = encoded["input_ids"].shape[1]
            torch.manual_seed(args.seed + repeat * 100000 + start)
            with torch.no_grad():
                sampled = model.generate(
                    **encoded, max_new_tokens=128, do_sample=True,
                    temperature=0.8, top_p=0.95, pad_token_id=tokenizer.pad_token_id,
                )
            for row, ids in zip(batch, sampled):
                raw = tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
                reward, details = rl.reward_response(row, raw)
                records.append({
                    "method": args.method,
                    "condition_id": row["condition_id"],
                    "sample_id": row["sample_id"],
                    "bucket": rl.target_bucket(row),
                    "mode": row["task_mode"],
                    "repeat": repeat,
                    "raw": raw,
                    "reward": reward,
                    **details,
                })
            print(
                f"[p25-gate:{args.method}] repeat={repeat + 1}/{args.repeats} "
                f"rows={min(start + batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )

    by_bucket: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_bucket[str(record["bucket"])].append(record)
    bucket_metrics = {}
    for bucket in rl.TARGET_BUCKETS:
        items = by_bucket[bucket]
        condition_hits: dict[str, list[bool]] = defaultdict(list)
        for item in items:
            condition_hits[str(item["condition_id"])].append(bool(item["strict"]))
        bucket_metrics[bucket] = {
            "candidates": len(items),
            "valid_rate": mean(bool(item["valid"]) for item in items),
            "property_strict_rate": mean(bool(item["property_strict"]) for item in items),
            "strict_rate": mean(bool(item["strict"]) for item in items),
            "relaxed_rate": mean(bool(item["relaxed"]) for item in items),
            "condition_hit_at_repeats": mean(any(values) for values in condition_hits.values()),
        }
    denovo = [bucket_metrics[f"de_novo:{count}p"] for count in (5, 6, 7)]
    edit = [bucket_metrics[f"edit:{task}"] for task in sorted(rl.TARGET_EDIT_TASKS)]
    summary = {
        "protocol": "p25_frozen_joint_gate_eval_v1",
        "method": args.method,
        "adapter": str(args.adapter_dir),
        "rows": len(rows),
        "repeats": args.repeats,
        "candidates": len(records),
        "sampling": {"temperature": 0.8, "top_p": 0.95, "seed": args.seed},
        "property_reranking": False,
        "target_access": False,
        "aggregate": {
            "de_novo_valid_macro": mean(item["valid_rate"] for item in denovo),
            "de_novo_strict_macro": mean(item["strict_rate"] for item in denovo),
            "de_novo_hit_at_repeats_macro": mean(item["condition_hit_at_repeats"] for item in denovo),
            "edit_valid_macro": mean(item["valid_rate"] for item in edit),
            "edit_strict_065_macro": mean(item["strict_rate"] for item in edit),
            "edit_relaxed_015_macro": mean(item["relaxed_rate"] for item in edit),
        },
        "buckets": bucket_metrics,
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
