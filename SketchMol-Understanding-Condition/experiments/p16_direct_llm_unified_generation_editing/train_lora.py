#!/usr/bin/env python3
"""Train one P16 LoRA arm on the cached full Qwen2.5-VL-7B base."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "unified_constraint_agent"
sys.path.insert(0, str(COMMON_DIR))
import train_common_llm_lora as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--arm", choices=("mixed", "denovo", "edit"), required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--seed", type=int, default=1616)
    args = parser.parse_args()

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("P16 requires BF16 CUDA")
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
    model = loader.from_pretrained(args.base_model, config=config, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = peft.get_peft_model(model, peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    dataset = common.ChatDataset(common.read_jsonl(args.train_jsonl), tokenizer, args.max_length)
    training_args = argparse.Namespace(
        output_dir=args.output_dir, epochs=args.epochs, batch_size=1,
        gradient_accumulation=args.gradient_accumulation, learning_rate=args.learning_rate,
        logging_steps=4, seed=args.seed,
    )
    trainer = transformers.Trainer(
        model=model,
        args=common.training_arguments(transformers, training_args, compute_dtype="bfloat16"),
        train_dataset=dataset,
        data_collator=common.CompletionCollator(tokenizer),
    )
    result = trainer.train()
    nonfinite = common.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"non-finite LoRA tensors: {nonfinite}")
    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    summary = {
        "protocol": "p16_direct_lora_sft_v1", "arm": args.arm, "base_model": args.base_model,
        "loader_kind": loader_kind, "vision_inputs_used": False, "train_rows": len(dataset),
        "epochs": args.epochs, "adapter_nonfinite_parameters": nonfinite,
        "train_metrics": dict(result.metrics),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
