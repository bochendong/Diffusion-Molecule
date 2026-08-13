#!/usr/bin/env python3
"""Evaluate held-out feedback-state action accuracy and common-task retention."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_common_llm_constrained_actions as constrained  # noqa: E402
import feedback_repair_agent_protocol as feedback  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-validation-jsonl", required=True, type=Path)
    parser.add_argument("--common-validation-jsonl", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def load_model(args: argparse.Namespace):
    import peft
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = peft.PeftModel.from_pretrained(model, args.adapter_dir).cuda().eval()
    return model, tokenizer


def mean(values: Sequence[bool]) -> float:
    return sum(bool(value) for value in values) / max(len(values), 1)


def score_actions(
    model: object,
    tokenizer: object,
    messages: Sequence[Mapping[str, object]],
    actions: Sequence[Mapping[str, object]],
    *,
    max_length: int,
    batch_size: int,
) -> list[float]:
    encoded = [
        constrained.encoded_action(tokenizer, messages, action, max_length=int(max_length))
        for action in actions
    ]
    return constrained.score_encoded_actions(model, tokenizer, encoded, batch_size=int(batch_size))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model, tokenizer = load_model(args)
    feedback_rows = protocol.read_jsonl(args.feedback_validation_jsonl)
    feedback_results = []
    state_counts: Counter[str] = Counter()
    for row in feedback_rows:
        messages = list(row["messages"])
        expected = json.loads(str(messages[-1]["content"]))
        user = json.loads(str(messages[-2]["content"]))
        actions = list(user["available_actions"])
        scores = score_actions(
            model,
            tokenizer,
            messages[:-1],
            actions,
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
        )
        selected = actions[max(range(len(scores)), key=scores.__getitem__)]
        correct = selected == expected
        state_type = str(row.get("state_type", "unknown"))
        state_counts[state_type] += 1
        feedback_results.append((state_type, correct))

    common_rows = protocol.read_jsonl(args.common_validation_jsonl)
    common_results = []
    for row in common_rows:
        messages = list(row["messages"])
        expected = json.loads(str(messages[-1]["content"]))
        score = score_actions(
            model,
            tokenizer,
            messages[:-1],
            [expected],
            max_length=int(args.max_length),
            batch_size=1,
        )[0]
        common_results.append((str(row.get("origin", "unknown")), float(score)))
    result = {
        "protocol": "feedback_repair_controller_validation_v1",
        "feedback": {
            "rows": len(feedback_results),
            "top1_action_accuracy": mean([item[1] for item in feedback_results]),
            "by_state": {
                state: {
                    "rows": state_counts[state],
                    "top1_action_accuracy": mean(
                        [correct for name, correct in feedback_results if name == state]
                    ),
                }
                for state in sorted(state_counts)
            },
        },
        "common_retention": {
            "rows": len(common_results),
            "mean_canonical_action_log_probability": sum(item[1] for item in common_results)
            / max(len(common_results), 1),
            "by_origin": {
                origin: {
                    "rows": sum(name == origin for name, _score in common_results),
                    "mean_canonical_action_log_probability": sum(
                        score for name, score in common_results if name == origin
                    )
                    / max(sum(name == origin for name, _score in common_results), 1),
                }
                for origin in sorted({name for name, _score in common_results})
            },
        },
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
    }
    protocol.write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
