#!/usr/bin/env python3
"""Evaluate one P33 arm on the frozen small Raw@1 gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
if str(P25_DIR) not in sys.path:
    sys.path.insert(0, str(P25_DIR))
import train_p23_joint_grpo as p25  # noqa: E402


def mean(values) -> float:
    values = list(values)
    return sum(float(value) for value in values) / max(len(values), 1)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def property_count(row: Mapping[str, object]) -> int:
    return p25.property_count(row)


def generate_records(model, tokenizer, rows, *, seed: int, batch_size: int, mode_offset: int):
    import torch

    records: list[dict[str, object]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True
            )
            for row in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        offset = encoded["input_ids"].shape[1]
        torch.manual_seed(seed + mode_offset + start)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row, ids in zip(batch, generated):
            raw = tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
            reward, details = p25.reward_response(row, raw)
            records.append(
                {
                    "condition_id": row.get("condition_id", row.get("sample_id", "")),
                    "task_mode": row["task_mode"],
                    "property_count": property_count(row),
                    "task_key": row.get("task_key", ""),
                    "raw": raw,
                    "reward": reward,
                    **details,
                }
            )
        print(f"[p33-eval] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo-gate", required=True, type=Path)
    parser.add_argument("--edit-gate", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm", choices=("joint", "denovo", "edit"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33051)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    de_novo = read_jsonl(args.denovo_gate)
    editing = [row for row in read_jsonl(args.edit_gate) if row["task_mode"] == "edit"]
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
    records = generate_records(
        model, tokenizer, de_novo,
        seed=args.seed, batch_size=args.batch_size, mode_offset=0,
    )
    records.extend(
        generate_records(
            model, tokenizer, editing,
            seed=args.seed, batch_size=args.batch_size, mode_offset=100000,
        )
    )

    de_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    edit_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record["task_mode"] == "de_novo":
            de_groups[int(record["property_count"])].append(record)
        else:
            edit_groups[str(record["task_key"])].append(record)
    de_buckets = {
        f"{count}p": {
            "rows": len(de_groups[count]),
            "strict_rate": mean(item["strict"] for item in de_groups[count]),
            "valid_rate": mean(item["valid"] for item in de_groups[count]),
        }
        for count in range(2, 8)
    }
    edit_buckets = {}
    for task in sorted(p25.TARGET_EDIT_TASKS):
        items = edit_groups[task]
        if not items:
            raise ValueError(f"missing P33 edit gate task {task}")
        edit_buckets[task] = {
            "rows": len(items),
            "strict_rate": mean(item["strict"] for item in items),
            "relaxed_rate": mean(item["relaxed"] for item in items),
            "valid_rate": mean(item["valid"] for item in items),
        }
    summary = {
        "protocol": "p33_clean_joint_vs_separate_raw1_pilot_v1",
        "arm": args.arm,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "seed": args.seed},
        "property_reranking": False,
        "rows": {"de_novo": len(de_novo), "edit": len(editing)},
        "aggregate": {
            "denovo_strict_macro": mean(v["strict_rate"] for v in de_buckets.values()),
            "denovo_valid_macro": mean(v["valid_rate"] for v in de_buckets.values()),
            "edit_strict_065_macro": mean(v["strict_rate"] for v in edit_buckets.values()),
            "edit_relaxed_015_macro": mean(v["relaxed_rate"] for v in edit_buckets.values()),
            "edit_valid_macro": mean(v["valid_rate"] for v in edit_buckets.values()),
        },
        "denovo_buckets": de_buckets,
        "edit_buckets": edit_buckets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "EVAL_COMPLETE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
