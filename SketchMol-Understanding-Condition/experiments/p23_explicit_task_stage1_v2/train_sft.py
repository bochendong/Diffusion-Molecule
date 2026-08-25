#!/usr/bin/env python3
"""Continue the P16 adapter on the expanded, explicitly conditioned P23 positives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "unified_constraint_agent"
sys.path.insert(0, str(COMMON_DIR))
import train_common_llm_lora as common  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2323)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("P23 requires BF16 CUDA")
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
        loader_kind = "causal_lm"
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
        loader_kind = "image_text_to_text_text_only"
    else:
        raise TypeError(f"unsupported config: {type(config).__name__}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.input_adapter, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    raw_rows = common.read_jsonl(args.train_jsonl)
    dataset = common.ChatDataset(raw_rows, tokenizer, args.max_length)
    training_ns = argparse.Namespace(
        output_dir=args.output_dir, epochs=args.epochs, batch_size=1,
        gradient_accumulation=args.gradient_accumulation, learning_rate=args.learning_rate,
        logging_steps=20, seed=args.seed,
    )
    trainer = transformers.Trainer(
        model=model,
        args=common.training_arguments(transformers, training_ns, compute_dtype="bfloat16"),
        train_dataset=dataset,
        data_collator=common.CompletionCollator(tokenizer),
    )
    result = trainer.train()
    nonfinite = common.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"non-finite trainable adapter parameters: {nonfinite}")
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    summary = {
        "protocol": "p23_explicit_task_positive_sft_stage1_v2",
        "base_model": args.base_model, "input_adapter": str(args.input_adapter),
        "loader_kind": loader_kind, "vision_inputs_used": False,
        "train_rows": len(dataset),
        "train_rows_by_mode": {
            mode: sum(row.get("task_mode") == mode for row in raw_rows)
            for mode in ("de_novo", "edit")
        },
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "mode_weighting": "1:1", "adapter_nonfinite_parameters": nonfinite,
        "train_metrics": dict(result.metrics), "adapter": str(adapter),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
