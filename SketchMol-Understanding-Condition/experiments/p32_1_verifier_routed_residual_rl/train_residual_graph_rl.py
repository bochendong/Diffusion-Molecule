#!/usr/bin/env python3
"""Continue one shared P32 policy with paired failed-proposal residual RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
UCA_DIR = SCRIPT_DIR.parent / "unified_constraint_agent"
for path in (SCRIPT_DIR, UCA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import residual_protocol as protocol  # noqa: E402
import train_common_llm_preference as common_train  # noqa: E402
import train_common_llm_tool_policy_grpo as common_rl  # noqa: E402


def stable_key(row: Mapping[str, object], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{row.get('example_id', '')}".encode()).hexdigest()


def select_failed(rows, mode: str, count: int, seed: int):
    grouped = defaultdict(list)
    for row in rows:
        if row["task_mode"] == mode and not bool(row["direct_details"].get("strict")):
            grouped[str(row["bucket"])].append(row)
    for bucket in grouped:
        grouped[bucket].sort(key=lambda row: stable_key(row, seed))
    buckets = sorted(grouped)
    cursors = {bucket: 0 for bucket in buckets}
    selected = []
    while len(selected) < count:
        progressed = False
        order = list(buckets)
        random.Random(seed + len(selected)).shuffle(order)
        for bucket in order:
            cursor = cursors[bucket]
            if cursor >= len(grouped[bucket]):
                continue
            selected.append(grouped[bucket][cursor])
            cursors[bucket] += 1
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    if len(selected) < count:
        raise ValueError(f"{mode}: only {len(selected)} direct-failed rows for {count} updates")
    random.Random(seed + 17).shuffle(selected)
    return selected


def save_checkpoint(model, tokenizer, output_dir: Path, step: int, history) -> Path:
    checkpoint = output_dir / f"checkpoint-{step:03d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint / "adapter")
    tokenizer.save_pretrained(checkpoint / "adapter")
    (checkpoint / "state.json").write_text(json.dumps({
        "protocol": protocol.PROTOCOL,
        "paired_updates": step,
        "history": history,
    }, indent=2, sort_keys=True) + "\n")
    (checkpoint / "CHECKPOINT_COMPLETE").touch()
    return checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--site-limit", type=int, default=24)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", default="10,20,30")
    parser.add_argument("--seed", type=int, default=32101)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    rows = protocol.read_jsonl(args.train_jsonl)
    by_mode = {
        mode: select_failed(rows, mode, args.pairs, args.seed + offset)
        for offset, mode in enumerate(("de_novo", "edit"))
    }
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
    model = peft.PeftModel.from_pretrained(base, args.input_adapter, is_trainable=True).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    checkpoints = {int(value) for value in args.checkpoint_steps.split(",") if value}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    helper_args = argparse.Namespace(
        temperature=args.temperature,
        max_length=args.max_length,
        gradient_accumulation=1,
    )

    for pair_index, (denovo, edit) in enumerate(
        zip(by_mode["de_novo"], by_mode["edit"]), start=1
    ):
        records = {"de_novo": denovo, "edit": edit}
        supports_by_mode = {}
        rollout_final = {}
        model.eval()
        for offset, mode in enumerate(("de_novo", "edit")):
            supports, final_smiles = protocol.online_supports(
                model,
                tokenizer,
                records[mode],
                max_steps=args.max_steps,
                max_actions=args.max_actions,
                site_limit=args.site_limit,
                max_length=args.max_length,
                score_batch_size=args.score_batch_size,
                temperature=args.temperature,
                seed=args.seed + pair_index * 1009 + offset,
            )
            if not supports:
                raise RuntimeError(f"no failed-proposal support for {mode} pair {pair_index}")
            supports_by_mode[mode] = supports
            rollout_final[mode] = protocol.score_smiles(records[mode], final_smiles)

        gradients = {}
        diagnostics = {}
        for mode in ("de_novo", "edit"):
            optimizer.zero_grad(set_to_none=True)
            model.train()
            loss, action_count, exact = common_rl.exact_action_value_backward(
                model,
                tokenizer,
                supports_by_mode[mode],
                args=helper_args,
                gradient_divisor=1,
            )
            gradients[mode] = common_rl.snapshot_parameter_gradients(parameters)
            diagnostics[mode] = {
                "example_id": records[mode]["example_id"],
                "supports": len(supports_by_mode[mode]),
                "actions": action_count,
                "loss": loss,
                "rollout_reward": rollout_final[mode].reward,
                "rollout_strict": rollout_final[mode].strict_success,
                **exact,
            }

        optimizer.zero_grad(set_to_none=True)
        merge = common_rl.assign_paired_pcgrad(
            parameters, gradients["de_novo"], gradients["edit"]
        )
        finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        if not finite:
            raise FloatingPointError(f"non-finite merged gradient at pair {pair_index}")
        norm_terms = [
            parameter.grad.float().pow(2).sum()
            for parameter in parameters if parameter.grad is not None
        ]
        unclipped = torch.sqrt(sum(norm_terms))
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        nonfinite = common_train.adapter_nonfinite_count(model)
        if nonfinite:
            raise FloatingPointError(f"adapter has {nonfinite} non-finite values")
        record = {
            "paired_update": pair_index,
            "by_mode": diagnostics,
            "unclipped_gradient_norm": float(unclipped),
            **merge,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if pair_index in checkpoints:
            save_checkpoint(model, tokenizer, args.output_dir, pair_index, history)

    final = args.output_dir / f"checkpoint-{args.pairs:03d}" / "adapter"
    if not final.is_dir():
        save_checkpoint(model, tokenizer, args.output_dir, args.pairs, history)
    adapter = args.output_dir / "adapter"
    if adapter.exists():
        shutil.rmtree(adapter)
    shutil.copytree(final, adapter)
    summary = {
        "protocol": protocol.PROTOCOL,
        "paired_updates": args.pairs,
        "gradient_conflicts": sum(bool(row["pcgrad_projected"]) for row in history),
        "mean_gradient_cosine": mean(float(row["pcgrad_cosine"]) for row in history),
        "adapter": str(adapter),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "TRAIN_COMPLETE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
