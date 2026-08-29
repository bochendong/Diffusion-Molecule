#!/usr/bin/env python3
"""Evaluate one greedy candidate per frozen P30.1 condition."""

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


def property_count(row: Mapping[str, object]) -> int:
    return p25.property_count(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-jsonl", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--macro-delta-min", type=float, default=0.02)
    parser.add_argument("--valid-delta-min", type=float, default=-0.01)
    parser.add_argument("--bucket-delta-min", type=float, default=-0.10)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    rows = p25.read_jsonl(args.prompts_jsonl)
    baseline = json.loads(args.baseline_summary.read_text())
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
    for start in range(0, len(rows), max(1, args.batch_size)):
        batch = rows[start : start + max(1, args.batch_size)]
        prompts = [
            tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
            for row in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        offset = encoded["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **encoded, max_new_tokens=128, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row, ids in zip(batch, generated):
            raw = tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
            reward, details = p25.reward_response(row, raw)
            records.append({
                "condition_id": row["condition_id"],
                "sample_id": row["sample_id"],
                "property_count": property_count(row),
                "raw": raw,
                "reward": reward,
                **details,
            })
        print(f"[p30.1-small-raw1] {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(record["property_count"])].append(record)
    buckets = {}
    bucket_deltas = {}
    for count in range(2, 8):
        items = grouped[count]
        if not items:
            raise ValueError(f"missing {count}p records")
        key = f"{count}p"
        buckets[key] = {
            "conditions": len(items),
            "strict_rate": mean(bool(item["strict"]) for item in items),
            "valid_rate": mean(bool(item["valid"]) for item in items),
        }
        bucket_deltas[key] = buckets[key]["strict_rate"] - float(
            baseline["buckets"][key]["strict_rate"]
        )
    aggregate = {
        "strict_macro": mean(buckets[f"{count}p"]["strict_rate"] for count in range(2, 8)),
        "valid_macro": mean(buckets[f"{count}p"]["valid_rate"] for count in range(2, 8)),
    }
    deltas = {
        "strict_macro": aggregate["strict_macro"] - float(baseline["aggregate"]["strict_macro"]),
        "valid_macro": aggregate["valid_macro"] - float(baseline["aggregate"]["valid_macro"]),
        "strict_by_arity": bucket_deltas,
    }
    gates = {
        "strict_macro_gain_ge_threshold": deltas["strict_macro"] >= args.macro_delta_min,
        "validity_drop_within_threshold": deltas["valid_macro"] >= args.valid_delta_min,
        "no_large_arity_regression": min(bucket_deltas.values()) >= args.bucket_delta_min,
    }
    result = {
        "protocol": "p30_1_alignment_raw1_rl_small_gate_result_v1",
        "adapter": str(args.adapter_dir),
        "conditions": len(records),
        "decoding": "greedy",
        "property_reranking": False,
        "aggregate": aggregate,
        "buckets": buckets,
        "baseline": baseline,
        "deltas": deltas,
        "gates": gates,
        "decision": "RUN_FULL_BUDGET_CURVE" if all(gates.values()) else "STOP_AFTER_SMALL_GATE",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# P30.1 alignment-refresh Raw@1 small gate",
        "",
        f"- baseline macro Raw@1: {100 * float(baseline['aggregate']['strict_macro']):.1f}%",
        f"- RL macro Raw@1: {100 * aggregate['strict_macro']:.1f}%",
        f"- delta: {100 * deltas['strict_macro']:+.1f} pp",
        f"- validity delta: {100 * deltas['valid_macro']:+.1f} pp",
        f"- decision: {result['decision']}",
        "",
    ]
    (args.output_dir / "RESULT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

