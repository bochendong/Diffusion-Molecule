#!/usr/bin/env python3
"""Continue P16 with chosen CE and edit-only source-copy margin loss."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "unified_constraint_agent"
sys.path.insert(0, str(COMMON_DIR))
import train_common_llm_lora as common  # noqa: E402


def encode_completion(tokenizer, messages, max_length: int) -> dict[str, list[int]]:
    full = common.input_id_list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    prompt = common.input_id_list(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True))
    eos = tokenizer.eos_token_id
    if eos is not None and (not full or full[-1] != eos):
        full.append(int(eos))
    full = full[:max_length]
    mask_len = min(common.common_prefix_length(full, prompt), len(full))
    labels = [-100] * mask_len + full[mask_len:]
    if not any(value != -100 for value in labels):
        raise ValueError("completion was truncated away")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


class PairedDataset:
    def __init__(self, rows: Sequence[Mapping[str, object]], tokenizer, max_length: int):
        self.examples = []
        for row in rows:
            messages = list(row["messages"])
            chosen = encode_completion(tokenizer, messages, max_length)
            paired = bool(row.get("pairwise_enabled"))
            if paired:
                rejected_messages = [*messages[:-1], {"role": "assistant", "content": str(row["rejected_assistant"])}]
                rejected = encode_completion(tokenizer, rejected_messages, max_length)
            else:
                rejected = chosen
            self.examples.append({
                **chosen,
                "rejected_input_ids": rejected["input_ids"],
                "rejected_attention_mask": rejected["attention_mask"],
                "rejected_labels": rejected["labels"],
                "pairwise_mask": int(paired),
                "edit_weight": 1.25 if paired else 1.0,
            })
        if not self.examples:
            raise ValueError("no P17 tokenized rows")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class PairedCollator:
    def __init__(self, tokenizer):
        self.pad = int(tokenizer.pad_token_id)

    def _pad(self, rows, prefix: str):
        import torch
        ids_key, mask_key, label_key = f"{prefix}input_ids", f"{prefix}attention_mask", f"{prefix}labels"
        width = max(len(row[ids_key]) for row in rows)
        return {
            ids_key: torch.tensor([row[ids_key] + [self.pad] * (width - len(row[ids_key])) for row in rows], dtype=torch.long),
            mask_key: torch.tensor([row[mask_key] + [0] * (width - len(row[mask_key])) for row in rows], dtype=torch.long),
            label_key: torch.tensor([row[label_key] + [-100] * (width - len(row[label_key])) for row in rows], dtype=torch.long),
        }

    def __call__(self, rows):
        import torch
        batch = self._pad(rows, "")
        batch.update(self._pad(rows, "rejected_"))
        batch["pairwise_mask"] = torch.tensor([row["pairwise_mask"] for row in rows], dtype=torch.float32)
        batch["edit_weight"] = torch.tensor([row["edit_weight"] for row in rows], dtype=torch.float32)
        return batch


def per_example_nll(logits, labels):
    import torch
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1),
        ignore_index=-100, reduction="none",
    ).view_as(shift_labels)
    return (loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--pairwise-margin", type=float, default=0.20)
    parser.add_argument("--pairwise-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1717)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("P17 requires BF16 CUDA")
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
        args.base_model, config=config, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
    )
    model = peft.PeftModel.from_pretrained(base, args.input_adapter, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    rows = common.read_jsonl(args.train_jsonl)
    dataset = PairedDataset(rows, tokenizer, args.max_length)
    training_ns = argparse.Namespace(
        output_dir=args.output_dir, epochs=args.epochs, batch_size=1,
        gradient_accumulation=args.gradient_accumulation, learning_rate=args.learning_rate,
        logging_steps=4, seed=args.seed,
    )

    class ContrastiveTrainer(transformers.Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            pair_mask = inputs.pop("pairwise_mask")
            edit_weight = inputs.pop("edit_weight")
            rejected = {
                "input_ids": inputs.pop("rejected_input_ids"),
                "attention_mask": inputs.pop("rejected_attention_mask"),
                "labels": inputs.pop("rejected_labels"),
            }
            chosen_out = model(**inputs)
            chosen_nll = per_example_nll(chosen_out.logits, inputs["labels"])
            ce = (chosen_nll * edit_weight.to(chosen_nll.device)).sum() / edit_weight.sum().clamp_min(1).to(chosen_nll.device)
            active = pair_mask.to(chosen_nll.device).sum()
            if active.item() > 0:
                rejected_out = model(**rejected)
                rejected_nll = per_example_nll(rejected_out.logits, rejected["labels"])
                margin = torch.relu(args.pairwise_margin + chosen_nll - rejected_nll)
                pair_loss = (margin * pair_mask.to(margin.device)).sum() / active.to(margin.device)
            else:
                pair_loss = chosen_nll.new_zeros(())
            loss = ce + args.pairwise_weight * pair_loss
            return (loss, chosen_out) if return_outputs else loss

    trainer = ContrastiveTrainer(
        model=model,
        args=common.training_arguments(transformers, training_ns, compute_dtype="bfloat16"),
        train_dataset=dataset,
        data_collator=PairedCollator(tokenizer),
    )
    result = trainer.train()
    nonfinite = common.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"non-finite trainable adapter parameters: {nonfinite}")
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    summary = {
        "protocol": "p17_copy_contrastive_sft_v1", "base_model": args.base_model,
        "input_adapter": str(args.input_adapter), "loader_kind": loader_kind,
        "vision_inputs_used": False, "train_rows": len(dataset),
        "edit_pair_rows": sum(bool(row.get("pairwise_enabled")) for row in rows),
        "denovo_rehearsal_rows": sum(row.get("task_mode") == "de_novo" for row in rows),
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "pairwise_margin": args.pairwise_margin, "pairwise_weight": args.pairwise_weight,
        "preference_optimizer": False, "adapter_nonfinite_parameters": nonfinite,
        "train_metrics": dict(result.metrics), "adapter": str(adapter),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
