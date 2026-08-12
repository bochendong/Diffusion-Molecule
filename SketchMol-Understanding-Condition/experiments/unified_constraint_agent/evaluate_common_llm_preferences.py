#!/usr/bin/env python3
"""Evaluate chosen-vs-rejected likelihood on frozen residual preferences."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_common_llm_constrained_actions as constrained  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import peft
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("Preference evaluation requires CUDA")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = False
    rows = read_jsonl(args.input_jsonl)
    encoded = []
    for row in rows:
        prompt = row.get("prompt_messages")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not isinstance(prompt, list) or not isinstance(chosen, Mapping) or not isinstance(rejected, Mapping):
            continue
        encoded.append(
            (
                constrained.encoded_action(tokenizer, prompt, chosen, max_length=args.max_length),
                constrained.encoded_action(tokenizer, prompt, rejected, max_length=args.max_length),
            )
        )
    flattened = [item for pair in encoded for item in pair]
    scores = constrained.score_encoded_actions(
        model,
        tokenizer,
        flattened,
        batch_size=int(args.batch_size),
    )
    margins = [scores[index] - scores[index + 1] for index in range(0, len(scores), 2)]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, margin in zip(rows, margins):
        grouped[str(row.get("preference_family") or row.get("origin") or "unknown")].append(margin)
    nonfinite = sum(not math.isfinite(value) for value in scores)
    summary = {
        "protocol": "common_llm_residual_preference_eval_v1",
        "variant": args.variant,
        "pairs": len(margins),
        "ranking_accuracy": sum(value > 0.0 for value in margins) / max(len(margins), 1),
        "mean_log_probability_margin": sum(margins) / max(len(margins), 1),
        "nonfinite_scores": nonfinite,
        "groups": {
            name: {
                "pairs": len(values),
                "ranking_accuracy": sum(value > 0.0 for value in values) / max(len(values), 1),
                "mean_log_probability_margin": sum(values) / max(len(values), 1),
            }
            for name, values in sorted(grouped.items())
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
