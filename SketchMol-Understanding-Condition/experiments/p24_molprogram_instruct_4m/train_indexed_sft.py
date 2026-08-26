#!/usr/bin/env python3
"""Memory-bounded continuation SFT over indexed MolProgramInstruct shards."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import struct
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "unified_constraint_agent"
sys.path.insert(0, str(COMMON_DIR))
import train_common_llm_lora as common  # noqa: E402


class IndexedChatDataset:
    """Map-style JSONL dataset backed by uint64 byte-offset sidecars."""

    def __init__(self, release_root: Path, tokenizer: object, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.paths: list[Path] = []
        self.indices: list[object] = []
        self.cumulative = [0]
        self._handles: dict[str, object] = {}
        for mode in ("de_novo", "edit"):
            for path in sorted((release_root / mode).glob("*.jsonl")):
                index = path.with_suffix(".idx")
                if not index.is_file() or index.stat().st_size % 8:
                    raise ValueError(f"missing or invalid index: {index}")
                import numpy as np

                offsets = np.memmap(index, dtype="<u8", mode="r")
                self.paths.append(path)
                self.indices.append(offsets)
                self.cumulative.append(self.cumulative[-1] + len(offsets))
        if not self.paths or not self.cumulative[-1]:
            raise ValueError(f"no indexed release shards under {release_root}")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _row(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.cumulative, index) - 1
        local_index = index - self.cumulative[shard]
        path = self.paths[shard]
        key = f"{os.getpid()}:{path}"
        handle = self._handles.get(key)
        if handle is None:
            handle = path.open("rb")
            self._handles[key] = handle
        handle.seek(int(self.indices[shard][local_index]))
        return json.loads(handle.readline())

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self._row(index)
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"invalid messages at dataset index {index}")
        full_ids = common.input_id_list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
            )
        )
        prompt_ids = common.input_id_list(
            self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True,
            )
        )
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
            full_ids.append(int(eos_id))
        full_ids = full_ids[: self.max_length]
        mask_length = min(common.common_prefix_length(full_ids, prompt_ids), len(full_ids))
        labels = [-100] * mask_length + full_ids[mask_length:]
        if not any(label != -100 for label in labels):
            raise ValueError(f"assistant target truncated at dataset index {index}")
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=24003)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args(argv)

    import inspect
    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("P24 requires BF16 CUDA")
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

    dataset = IndexedChatDataset(args.release_root, tokenizer, args.max_length)
    values = {
        "output_dir": str(args.output_dir), "max_steps": args.max_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate, "warmup_steps": args.warmup_steps,
        "weight_decay": 0.01, "max_grad_norm": 1.0,
        "logging_steps": args.logging_steps, "save_strategy": "steps",
        "save_steps": args.save_steps, "save_total_limit": 2,
        "bf16": True, "fp16": False, "gradient_checkpointing": True,
        "optim": "adamw_torch", "report_to": [],
        "logging_nan_inf_filter": False, "remove_unused_columns": False,
        "seed": args.seed, "data_seed": args.seed,
    }
    signature = inspect.signature(transformers.TrainingArguments.__init__)
    training_args = transformers.TrainingArguments(
        **{key: value for key, value in values.items() if key in signature.parameters}
    )
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=dataset,
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
        "protocol": "p24_molprogram_instruct_4m_indexed_sft_v1",
        "base_model": args.base_model, "input_adapter": str(args.input_adapter),
        "loader_kind": loader_kind, "train_rows": len(dataset),
        "max_steps": args.max_steps, "gradient_accumulation": args.gradient_accumulation,
        "effective_examples": args.max_steps * args.gradient_accumulation,
        "learning_rate": args.learning_rate, "resume_checkpoint": resume,
        "adapter_nonfinite_parameters": nonfinite,
        "train_metrics": dict(result.metrics), "adapter": str(adapter),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "TRAINING_COMPLETE").write_text(
        hashlib_sha256(adapter / "adapter_model.safetensors") + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def hashlib_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
