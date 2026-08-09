#!/usr/bin/env python3
"""Reference-free pairwise preference tuning for executable actions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_common_llm_constrained_actions as constrained  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", type=Path, default=None)
    parser.add_argument("--input-adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--sft-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1704)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--replay-jsonl", type=Path, default=None)
    parser.add_argument("--replay-sft-weight", type=float, default=0.0)
    parser.add_argument("--replay-batch-size", type=int, default=1)
    parser.add_argument("--replay-max-per-origin", type=int, default=256)
    parser.add_argument("--replay-origins", default="denovo,table1,mumo")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class PairDataset:
    def __init__(self, rows: Sequence[Mapping[str, object]], tokenizer: object, max_length: int):
        self.examples = []
        for row in rows:
            prompt = row.get("prompt_messages")
            chosen = row.get("chosen")
            rejected = row.get("rejected")
            if not isinstance(prompt, list) or not isinstance(chosen, Mapping) or not isinstance(rejected, Mapping):
                continue
            self.examples.append(
                {
                    "chosen": constrained.encoded_action(
                        tokenizer,
                        prompt,
                        chosen,
                        max_length=max_length,
                    ),
                    "rejected": constrained.encoded_action(
                        tokenizer,
                        prompt,
                        rejected,
                        max_length=max_length,
                    ),
                }
            )
        if not self.examples:
            raise ValueError("No tokenized preference pairs were produced")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        return self.examples[index]


class PairCollator:
    def __init__(self, tokenizer: object):
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
        import torch

        flattened = [feature[kind] for feature in features for kind in ("chosen", "rejected")]
        width = max(len(item["input_ids"]) for item in flattened)
        input_ids = []
        attention_mask = []
        labels = []
        for item in flattened:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def task_balanced_replay_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    origins: Sequence[str],
    max_per_origin: int,
    seed: int,
) -> list[Mapping[str, object]]:
    requested = {str(origin).strip() for origin in origins if str(origin).strip()}
    grouped: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for index, row in enumerate(rows):
        origin = str(row.get("origin", "") or "").strip()
        if requested and origin not in requested:
            continue
        identity = str(row.get("example_id", "") or f"{origin}:{index}")
        grouped[origin].setdefault(identity, row)
    missing = sorted(requested - set(grouped))
    if missing:
        raise ValueError(f"Anti-forgetting replay is missing requested origins: {missing}")
    per_origin = min(
        max(1, int(max_per_origin)),
        min((len(items) for items in grouped.values()), default=0),
    )
    if per_origin <= 0:
        raise ValueError("Anti-forgetting replay contains no eligible rows")
    output: list[Mapping[str, object]] = []
    for offset, origin in enumerate(sorted(grouped)):
        items = list(grouped[origin].values())
        random.Random(int(seed) + offset).shuffle(items)
        output.extend(items[:per_origin])
    random.Random(int(seed)).shuffle(output)
    return output


class ReplayDataset:
    def __init__(self, rows: Sequence[Mapping[str, object]], tokenizer: object, max_length: int):
        self.examples = []
        self.origin_counts = Counter()
        for row in rows:
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            assistant = messages[-1]
            if not isinstance(assistant, Mapping) or str(assistant.get("role", "")) != "assistant":
                continue
            try:
                payload = json.loads(str(assistant.get("content", "") or ""))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, Mapping):
                continue
            self.examples.append(
                constrained.encoded_action(
                    tokenizer,
                    messages[:-1],
                    payload,
                    max_length=max_length,
                )
            )
            self.origin_counts[str(row.get("origin", "") or "unknown")] += 1
        if not self.examples:
            raise ValueError("No tokenized anti-forgetting replay rows were produced")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        return self.examples[index]


class ReplayCollator:
    def __init__(self, tokenizer: object):
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
        import torch

        width = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def sequence_mean_log_probability(model: object, batch: Mapping[str, object]):
    import torch.nn.functional as functional

    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    token_losses = functional.cross_entropy(
        logits.transpose(1, 2),
        labels,
        reduction="none",
        ignore_index=-100,
    )
    mask = labels.ne(-100)
    return -(token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def adapter_nonfinite_count(model: object) -> int:
    import torch

    return sum(
        int((~torch.isfinite(parameter.detach().float())).sum().item())
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import peft
        import torch
        import torch.nn.functional as functional
        import transformers
    except ImportError as exc:
        raise SystemExit(f"Missing common-LLM preference dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("Common-LLM preference training requires CUDA")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = peft.PeftModel.from_pretrained(
        base,
        args.input_adapter_dir,
        is_trainable=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model = model.cuda().train()

    dataset = PairDataset(read_jsonl(args.train_jsonl), tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=PairCollator(tokenizer),
    )
    replay_dataset = None
    replay_loader = None
    replay_iterator = None
    if args.replay_jsonl is not None and float(args.replay_sft_weight) > 0.0:
        replay_origins = [item.strip() for item in str(args.replay_origins).split(",") if item.strip()]
        replay_rows = task_balanced_replay_rows(
            read_jsonl(args.replay_jsonl),
            origins=replay_origins,
            max_per_origin=int(args.replay_max_per_origin),
            seed=int(args.seed),
        )
        replay_dataset = ReplayDataset(replay_rows, tokenizer, args.max_length)
        replay_loader = torch.utils.data.DataLoader(
            replay_dataset,
            batch_size=int(args.replay_batch_size),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(args.seed) + 1),
            collate_fn=ReplayCollator(tokenizer),
        )
        replay_iterator = iter(replay_loader)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    history = []
    global_step = 0
    update_step = 0
    for epoch in range(1, int(args.epochs) + 1):
        running_loss = 0.0
        running_pair = 0.0
        running_replay = 0.0
        running_accuracy = 0.0
        seen = 0
        replay_seen = 0
        for batch_index, batch in enumerate(loader, start=1):
            batch = {key: value.cuda() for key, value in batch.items()}
            scores = sequence_mean_log_probability(model, batch)
            chosen = scores[0::2]
            rejected = scores[1::2]
            margin = chosen - rejected
            pair_loss = functional.softplus(-float(args.beta) * margin).mean()
            sft_loss = -chosen.mean()
            loss = pair_loss + float(args.sft_weight) * sft_loss
            replay_loss = None
            if replay_loader is not None:
                assert replay_iterator is not None
                try:
                    replay_batch = next(replay_iterator)
                except StopIteration:
                    replay_iterator = iter(replay_loader)
                    replay_batch = next(replay_iterator)
                replay_batch = {key: value.cuda() for key, value in replay_batch.items()}
                replay_scores = sequence_mean_log_probability(model, replay_batch)
                replay_loss = -replay_scores.mean()
                loss = loss + float(args.replay_sft_weight) * replay_loss
                replay_seen += int(replay_scores.numel())
            (loss / int(args.gradient_accumulation)).backward()
            global_step += 1
            do_update = batch_index % int(args.gradient_accumulation) == 0 or batch_index == len(loader)
            if do_update:
                torch.nn.utils.clip_grad_norm_(parameters, float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if update_step <= 5 or update_step % 10 == 0:
                    nonfinite = adapter_nonfinite_count(model)
                    if nonfinite:
                        raise FloatingPointError(
                            f"Detected {nonfinite} non-finite preference adapter parameters at update {update_step}"
                        )
            batch_pairs = int(chosen.numel())
            seen += batch_pairs
            running_loss += float(loss.detach()) * batch_pairs
            running_pair += float(pair_loss.detach()) * batch_pairs
            if replay_loss is not None:
                running_replay += float(replay_loss.detach()) * batch_pairs
            running_accuracy += float((margin.detach() > 0).float().sum())
            if global_step % int(args.logging_steps) == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "pairs": seen,
                            "loss": running_loss / max(seen, 1),
                            "pair_loss": running_pair / max(seen, 1),
                            "replay_sft_loss": running_replay / max(seen, 1) if replay_loader else None,
                            "ranking_accuracy": running_accuracy / max(seen, 1),
                            "replay_examples": replay_seen,
                            "updates": update_step,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch,
                "pairs": seen,
                "loss": running_loss / max(seen, 1),
                "pair_loss": running_pair / max(seen, 1),
                "replay_sft_loss": running_replay / max(seen, 1) if replay_loader else None,
                "ranking_accuracy": running_accuracy / max(seen, 1),
                "replay_examples": replay_seen,
                "updates": update_step,
            }
        )

    nonfinite = adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"Refusing to save preference adapter with {nonfinite} non-finite parameters")
    adapter_dir = args.output_dir / "adapter"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    summary = {
        "protocol": "unified_constraint_common_llm_preference_v1",
        "base_model": args.base_model,
        "input_adapter_dir": str(args.input_adapter_dir),
        "train_jsonl": str(args.train_jsonl),
        "validation_jsonl": str(args.validation_jsonl) if args.validation_jsonl else None,
        "train_pairs": len(dataset),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "sft_weight": args.sft_weight,
        "gradient_accumulation": args.gradient_accumulation,
        "replay_jsonl": str(args.replay_jsonl) if args.replay_jsonl else None,
        "replay_sft_weight": args.replay_sft_weight,
        "replay_rows": len(replay_dataset) if replay_dataset is not None else 0,
        "replay_origin_counts": (
            dict(sorted(replay_dataset.origin_counts.items())) if replay_dataset is not None else {}
        ),
        "adapter_nonfinite_parameters": nonfinite,
        "history": history,
        "adapter_dir": str(adapter_dir),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
