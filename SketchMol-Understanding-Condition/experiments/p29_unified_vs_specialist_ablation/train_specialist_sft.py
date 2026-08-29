#!/usr/bin/env python3
"""Train an exact mode-specific continuation of the frozen joint sampler."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
P24_DIR = SCRIPT_DIR.parent / "p24_molprogram_instruct_4m"
COMMON_DIR = SCRIPT_DIR.parent / "unified_constraint_agent"
for path in (P24_DIR, COMMON_DIR):
    sys.path.insert(0, str(path))

import train_common_llm_lora as common  # noqa: E402
import train_indexed_sft as p24  # noqa: E402


class ExactSpecialistSampler:
    """Reuse one mode's exact rows from the frozen thirteen-bucket sampler."""

    def __init__(
        self,
        dataset: object,
        seed: int,
        task_mode: str,
        rows_per_task: int,
        batch_size: int = 1,
    ):
        if task_mode not in {"de_novo", "edit"}:
            raise ValueError(f"specialist task mode must be de_novo or edit: {task_mode}")
        if rows_per_task < 1 or batch_size < 1 or rows_per_task % batch_size:
            raise ValueError("rows per task must be positive and divisible by physical batch size")
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.rows_per_task = rows_per_task
        self.all_keys = sorted(dataset.bucket_indices)
        self.keys = [key for key in self.all_keys if key.startswith(f"{task_mode}:")]
        expected_count = 6 if task_mode == "de_novo" else 7
        if len(self.keys) != expected_count:
            raise ValueError(
                f"expected {expected_count} {task_mode} buckets, found {self.keys}"
            )
        for key in self.all_keys:
            if len(dataset.bucket_indices[key]) < rows_per_task:
                raise ValueError(
                    f"bucket {key} has {len(dataset.bucket_indices[key])} rows; "
                    f"requires {rows_per_task}"
                )

    def __len__(self) -> int:
        return self.rows_per_task * len(self.keys)

    def __iter__(self):
        import torch

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # Generate all thirteen permutations in the same sorted order as the
        # frozen joint run, then retain only the specialist's active buckets.
        permutations = {
            key: torch.randperm(
                len(self.dataset.bucket_indices[key]), generator=generator,
            )[: self.rows_per_task]
            for key in self.all_keys
        }
        for start in range(0, self.rows_per_task, self.batch_size):
            for key in self.keys:
                for position in range(start, start + self.batch_size):
                    local = int(permutations[key][position])
                    yield int(self.dataset.bucket_indices[key][local])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--task-mode", required=True, choices=("de_novo", "edit"))
    parser.add_argument("--rows-per-task", type=int, default=81415)
    parser.add_argument("--expected-release-rows", type=int, default=2569919)
    parser.add_argument("--expected-examples", required=True, type=int)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=65)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=24003)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    if args.per_device_batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("batch size and gradient accumulation must be positive")

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("P29 specialist training requires BF16 CUDA")
    if not args.release_root.joinpath("RELEASE_COMPLETE").is_file():
        raise FileNotFoundError(f"release is not frozen: {args.release_root}")
    if not args.input_adapter.joinpath("adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"missing input adapter: {args.input_adapter}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
        loader_kind = "causal_lm"
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
        loader_kind = "image_text_to_text_text_only"
    else:
        raise TypeError(f"unsupported config: {type(config).__name__}")
    base = loader.from_pretrained(
        args.base_model,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(
        base, args.input_adapter, is_trainable=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    dataset = p24.IndexedChatDataset(args.release_root, tokenizer, args.max_length)
    if len(dataset) != args.expected_release_rows:
        raise ValueError(
            f"release rows differ: expected={args.expected_release_rows} actual={len(dataset)}"
        )
    sampler = ExactSpecialistSampler(
        dataset,
        seed=args.seed,
        task_mode=args.task_mode,
        rows_per_task=args.rows_per_task,
        batch_size=args.per_device_batch_size,
    )
    if len(sampler) != args.expected_examples:
        raise ValueError(
            f"specialist exposure differs: expected={args.expected_examples} actual={len(sampler)}"
        )
    optimizer_steps = math.ceil(
        math.ceil(len(sampler) / args.per_device_batch_size)
        / args.gradient_accumulation
    )
    values = {
        "output_dir": str(args.output_dir),
        "max_steps": -1,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "report_to": [],
        "logging_nan_inf_filter": False,
        "remove_unused_columns": False,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    signature = inspect.signature(transformers.TrainingArguments.__init__)
    training_args = transformers.TrainingArguments(
        **{key: value for key, value in values.items() if key in signature.parameters}
    )

    class SpecialistTrainer(transformers.Trainer):
        def _get_train_sampler(self, train_dataset=None):
            selected = train_dataset if train_dataset is not None else self.train_dataset
            return ExactSpecialistSampler(
                selected,
                seed=args.seed,
                task_mode=args.task_mode,
                rows_per_task=args.rows_per_task,
                batch_size=args.per_device_batch_size,
            )

    trainer = SpecialistTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=common.CompletionCollator(tokenizer),
    )
    resume: bool | str = False
    if args.resume_from_checkpoint:
        checkpoints = sorted(
            args.output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        resume = str(checkpoints[-1]) if checkpoints else False
    result = trainer.train(resume_from_checkpoint=resume)
    nonfinite = common.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"non-finite trainable adapter parameters: {nonfinite}")
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    summary = {
        "protocol": "p29_exact_specialist_continuation_v1",
        "base_model": args.base_model,
        "input_adapter": str(args.input_adapter),
        "loader_kind": loader_kind,
        "task_mode": args.task_mode,
        "active_task_buckets": sampler.keys,
        "rows_per_task": args.rows_per_task,
        "effective_examples": len(sampler),
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.per_device_batch_size * args.gradient_accumulation,
        "expected_optimizer_steps": optimizer_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "sampler_reuses_joint_permutations": True,
        "resume_checkpoint": resume,
        "adapter_nonfinite_parameters": nonfinite,
        "train_metrics": dict(result.metrics),
        "adapter": str(adapter),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "TRAINING_COMPLETE").write_text(
        p24.hashlib_sha256(adapter / "adapter_model.safetensors") + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
