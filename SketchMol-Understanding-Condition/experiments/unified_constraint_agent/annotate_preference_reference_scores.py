#!/usr/bin/env python3
"""Annotate preference pairs with frozen stable-adapter reference margins."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_common_llm_constrained_actions as constrained  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-train-jsonl", required=True, type=Path)
    parser.add_argument("--input-validation-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def annotate(
    rows: Sequence[Mapping[str, object]],
    *,
    model: object,
    tokenizer: object,
    batch_size: int,
    max_length: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    encoded = []
    eligible = []
    for row in rows:
        prompt = row.get("prompt_messages")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not isinstance(prompt, list) or not isinstance(chosen, Mapping) or not isinstance(rejected, Mapping):
            continue
        encoded.extend(
            [
                constrained.encoded_action(tokenizer, prompt, chosen, max_length=max_length),
                constrained.encoded_action(tokenizer, prompt, rejected, max_length=max_length),
            ]
        )
        eligible.append(dict(row))
    scores = constrained.score_encoded_actions(
        model, tokenizer, encoded, batch_size=max(1, int(batch_size))
    )
    output = []
    margins = []
    for index, row in enumerate(eligible):
        chosen_score = float(scores[2 * index])
        rejected_score = float(scores[2 * index + 1])
        margin = chosen_score - rejected_score
        if not all(math.isfinite(value) for value in (chosen_score, rejected_score, margin)):
            raise FloatingPointError("Non-finite stable reference score")
        row.update(
            {
                "stable_reference_chosen_log_probability": chosen_score,
                "stable_reference_rejected_log_probability": rejected_score,
                "stable_reference_margin": margin,
            }
        )
        output.append(row)
        margins.append(margin)
    return output, {
        "input_rows": len(rows),
        "annotated_rows": len(output),
        "reference_ranking_accuracy": sum(value > 0 for value in margins) / max(len(margins), 1),
        "mean_reference_margin": sum(margins) / max(len(margins), 1),
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import peft
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("Reference preference annotation requires CUDA")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = False
    train, train_summary = annotate(
        read_jsonl(args.input_train_jsonl),
        model=model,
        tokenizer=tokenizer,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    validation, validation_summary = annotate(
        read_jsonl(args.input_validation_jsonl),
        model=model,
        tokenizer=tokenizer,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    summary = {
        "protocol": "stable_reference_preference_annotation_v1",
        "adapter_dir": str(args.adapter_dir),
        "reference_margin_field": "stable_reference_margin",
        "train": train_summary,
        "validation": validation_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
