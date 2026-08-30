#!/usr/bin/env python3
"""Evaluate one frozen P32 action-policy checkpoint on both task modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import graph_repair_protocol as protocol  # noqa: E402


def mode_summary(rows, details_key: str):
    return {
        mode: protocol.aggregate_records(
            [row for row in rows if row["task_mode"] == mode], details_key
        )
        for mode in ("de_novo", "edit")
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--site-limit", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--score-batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True

    records = protocol.read_jsonl(args.gate_jsonl)
    evaluated = []
    for index, record in enumerate(records):
        final_smiles, trace = protocol.greedy_rollout(
            model,
            tokenizer,
            record,
            max_steps=args.max_steps,
            max_actions=args.max_actions,
            site_limit=args.site_limit,
            max_length=args.max_length,
            score_batch_size=args.score_batch_size,
        )
        feedback = protocol.score_smiles(record, final_smiles)
        evaluated.append({
            "protocol": protocol.PROTOCOL,
            "checkpoint": args.tag,
            "example_id": record["example_id"],
            "task_mode": record["task_mode"],
            "bucket": record["bucket"],
            "direct_smiles": record.get("direct_smiles", ""),
            "direct_details": record["direct_details"],
            "policy_smiles": final_smiles,
            "policy_details": feedback.details,
            "policy_reward": feedback.reward,
            "trace": trace,
        })
        if (index + 1) % 10 == 0:
            print(f"[p32-eval:{args.tag}] {index + 1}/{len(records)}", flush=True)

    direct = mode_summary(evaluated, "direct_details")
    policy = mode_summary(evaluated, "policy_details")
    deltas = {
        mode: {
            metric: float(policy[mode][metric]) - float(direct[mode][metric])
            for metric in ("strict_macro", "relaxed_macro", "valid_macro")
        }
        for mode in ("de_novo", "edit")
    }
    result = {
        "protocol": protocol.PROTOCOL,
        "checkpoint": args.tag,
        "adapter_dir": str(args.adapter_dir),
        "target_blind": True,
        "property_reranking": False,
        "direct": direct,
        "policy": policy,
        "delta_policy_minus_direct": deltas,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol.write_jsonl(args.output_dir / "candidates.jsonl", evaluated)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "EVAL_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
