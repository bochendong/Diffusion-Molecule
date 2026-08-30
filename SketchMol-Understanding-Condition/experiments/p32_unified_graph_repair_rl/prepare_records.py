#!/usr/bin/env python3
"""Freeze P32 train/gate rows and materialize P24 direct proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P30_DIR = SCRIPT_DIR.parent / "p30_balanced_shared_policy_rl"
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
for path in (SCRIPT_DIR, P30_DIR, P25_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import graph_repair_protocol as protocol  # noqa: E402
import train_balanced_shared_rl as p30  # noqa: E402
import train_p23_joint_grpo as p25  # noqa: E402


def stable_key(row: Mapping[str, object], seed: int) -> str:
    identity = row.get("example_id", row.get("sample_id", row.get("condition_id", "")))
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def select_balanced(rows, buckets, per_bucket: int, seed: int):
    grouped = defaultdict(list)
    for row in rows:
        bucket = p30.balanced_bucket(row)
        if bucket:
            grouped[bucket].append(row)
    output = []
    for bucket in buckets:
        values = sorted(grouped[bucket], key=lambda row: stable_key(row, seed))
        if len(values) < per_bucket:
            raise ValueError(f"{bucket}: {len(values)} < {per_bucket}")
        output.extend(values[:per_bucket])
    return output


def select_gate(denovo_rows, edit_rows, seed: int):
    grouped = defaultdict(list)
    for row in denovo_rows:
        grouped[p30.balanced_bucket(row)].append(row)
    for row in edit_rows:
        if str(row.get("task_mode", "")) == "edit":
            grouped[p30.balanced_bucket(row)].append(row)
    output = []
    for bucket in p30.DE_NOVO_BUCKETS:
        values = sorted(grouped[bucket], key=lambda row: stable_key(row, seed))
        if len(values) < 20:
            raise ValueError(f"gate {bucket}: {len(values)} < 20")
        output.extend(values[:20])
    for bucket in p30.EDIT_BUCKETS:
        values = sorted(grouped[bucket], key=lambda row: stable_key(row, seed + 1))
        if len(values) < 5:
            raise ValueError(f"gate {bucket}: {len(values)} < 5")
        output.extend(values[:5])
    return output


def ir_for_row(row: Mapping[str, object]) -> dict[str, object]:
    mode = str(row["task_mode"])
    payload = p25.prompt_payload(row)
    source = str(payload.get("source", "") or "")
    if source == "<EMPTY>":
        source = ""
    constraints = []
    for item in payload.get("conditions", []):
        if not isinstance(item, Mapping):
            continue
        prop = str(item.get("property", "") or "")
        goal = item.get("goal")
        constraint: dict[str, object] = {"property": prop, "hard": True}
        if isinstance(goal, Mapping) and "around" in goal:
            constraint.update({"objective": "target", "target": float(goal["around"])})
        else:
            direction = 1 if str(goal) == "increase" else -1 if str(goal) == "decrease" else 0
            constraint.update({"objective": "improve", "direction": direction})
        constraints.append(constraint)
    return {
        "protocol": protocol.PROTOCOL,
        "condition_id": str(row.get("condition_id", row.get("example_id", ""))),
        "task_mode": mode,
        "source_smiles": source,
        "constraints": constraints,
    }


def prompt_messages(row: Mapping[str, object]):
    messages = list(row["messages"])
    if messages and str(messages[-1].get("role")) == "assistant":
        messages = messages[:-1]
    return messages


def materialize(model, tokenizer, rows, *, batch_size: int):
    import torch

    output = []
    tokenizer.padding_side = "left"
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(prompt_messages(row), tokenize=False, add_generation_prompt=True)
            for row in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        offset = encoded["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **encoded, max_new_tokens=128, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row, ids in zip(batch, generated):
            raw = tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
            _reward, details = p25.reward_response(row, raw)
            parsed = p25.protocol.parse_response(raw, str(row["task_mode"]))
            direct = str(parsed.get("smiles", "") or "")
            ir = ir_for_row(row)
            mode = str(row["task_mode"])
            source = str(ir.get("source_smiles", "") or "")
            initial = direct if mode == "de_novo" and direct else source if mode == "edit" else "C"
            output.append({
                "protocol": protocol.PROTOCOL,
                "example_id": str(row.get("example_id", row.get("sample_id", row.get("condition_id", "")))),
                "task_mode": mode,
                "bucket": p30.balanced_bucket(row),
                "constraint_ir": ir,
                "benchmark_row": dict(row),
                "initial_smiles": initial,
                "direct_smiles": direct,
                "direct_raw": raw,
                "direct_details": details,
                "initial_fallback_used": bool(mode == "de_novo" and not direct),
            })
        print(f"[p32-prepare] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--denovo-gate-jsonl", required=True, type=Path)
    parser.add_argument("--edit-gate-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=32001)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    train_all = p25.read_jsonl(args.train_jsonl)
    train_rows = []
    for mode, per_bucket in (("de_novo", 5), ("edit", 3)):
        selected = select_balanced(
            [row for row in train_all if str(row.get("task_mode")) == mode],
            p30.DE_NOVO_BUCKETS if mode == "de_novo" else p30.EDIT_BUCKETS,
            per_bucket,
            args.seed + (0 if mode == "de_novo" else 1),
        )
        train_rows.extend(row for row in selected if str(row.get("task_mode")) == mode)
    gate_rows = select_gate(
        p25.read_jsonl(args.denovo_gate_jsonl),
        p25.read_jsonl(args.edit_gate_jsonl),
        args.seed + 2,
    )
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    loader = (
        transformers.AutoModelForCausalLM
        if type(config) in transformers.AutoModelForCausalLM._model_mapping
        else transformers.AutoModelForImageTextToText
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True
    prepared_train = materialize(model, tokenizer, train_rows, batch_size=args.batch_size)
    prepared_gate = materialize(model, tokenizer, gate_rows, batch_size=args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol.write_jsonl(args.output_dir / "train.jsonl", prepared_train)
    protocol.write_jsonl(args.output_dir / "gate.jsonl", prepared_gate)
    manifest = {
        "protocol": protocol.PROTOCOL,
        "train_rows": len(prepared_train),
        "gate_rows": len(prepared_gate),
        "train_by_mode": {
            mode: sum(row["task_mode"] == mode for row in prepared_train)
            for mode in ("de_novo", "edit")
        },
        "gate_by_mode": {
            mode: sum(row["task_mode"] == mode for row in prepared_gate)
            for mode in ("de_novo", "edit")
        },
        "fallbacks": sum(bool(row["initial_fallback_used"]) for row in [*prepared_train, *prepared_gate]),
        "target_molecules_used_for_policy_input": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
