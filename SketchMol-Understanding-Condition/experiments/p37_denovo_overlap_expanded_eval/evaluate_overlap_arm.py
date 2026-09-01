#!/usr/bin/env python3
"""Evaluate one frozen adapter on the P37 de novo overlap gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P33_DIR = SCRIPT_DIR.parent / "p33_joint_vs_separate_pilot"


SHARED_PROPERTIES = frozenset({"MW", "LogP", "QED", "HBA", "RB"})


def group_for_task_key(task_key: str) -> str:
    properties = {part.split(":", 1)[0] for part in task_key.split("+")}
    return "shared_only" if properties <= SHARED_PROPERTIES else "contains_denovo_only"


def mean(values) -> float:
    values = list(values)
    return sum(float(value) for value in values) / len(values)


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    cells: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        group = group_for_task_key(str(record["task_key"]))
        cells[(group, int(record["property_count"]))].append(record)
    buckets = {}
    for group in ("shared_only", "contains_denovo_only"):
        buckets[group] = {
            f"{arity}p": {
                "rows": len(cells[(group, arity)]),
                "strict_rate": mean(row["strict"] for row in cells[(group, arity)]),
                "valid_rate": mean(row["valid"] for row in cells[(group, arity)]),
            }
            for arity in (2, 3, 4, 5)
        }
    scopes = {}
    for name, arities in (("2p4p", (2, 3, 4)), ("2p5p", (2, 3, 4, 5))):
        scopes[name] = {
            group: {
                "rows": sum(len(cells[(group, arity)]) for arity in arities),
                "strict_arity_macro": mean(
                    mean(row["strict"] for row in cells[(group, arity)])
                    for arity in arities
                ),
                "valid_arity_macro": mean(
                    mean(row["valid"] for row in cells[(group, arity)])
                    for arity in arities
                ),
            }
            for group in ("shared_only", "contains_denovo_only")
        }
    return {"buckets": buckets, "scopes": scopes}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm", choices=("joint", "denovo"), required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=37151)
    args = parser.parse_args(argv)

    import peft
    import sys
    import torch
    import transformers

    if str(P33_DIR) not in sys.path:
        sys.path.insert(0, str(P33_DIR))
    import evaluate_arm as p33

    gate = p33.read_jsonl(args.gate)
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
        args.base_model,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True
    records = p33.generate_records(
        model, tokenizer, gate, seed=args.seed, batch_size=args.batch_size, mode_offset=0
    )
    result = {
        "protocol": "p37_denovo_overlap_expanded_raw1_v1",
        "scale": args.scale,
        "arm": args.arm,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "seed": args.seed},
        "candidate_budget": 1,
        "property_reranking": False,
        "rows": len(records),
        **summarize(records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "EVAL_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
