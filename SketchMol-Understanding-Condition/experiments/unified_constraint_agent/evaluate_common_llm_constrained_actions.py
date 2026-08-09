#!/usr/bin/env python3
"""Rank an n=20 executable GraphEditDSL pool with the common LLM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_DIR = SCRIPT_DIR.parent / "unified_smiles_generator"
for import_dir in (SCRIPT_DIR, POLICY_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


def policy_module():
    import umtp_graph_action_policy

    return umtp_graph_action_policy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--candidate-budget", type=int, default=20)
    parser.add_argument("--site-limit", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--table1-similarity-threshold", type=float, default=0.65)
    parser.add_argument("--mumo-similarity-threshold", type=float, default=0.40)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def constraint_payload(record: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    messages = record.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("A constrained-action record requires system, user, and assistant messages")
    user = json.loads(str(messages[-2]["content"]))
    expected = json.loads(str(messages[-1]["content"]))
    ir = user.get("constraint_ir", {})
    if not isinstance(ir, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("Malformed constraint or expected-action payload")
    return dict(ir), dict(expected)


def planner_row_from_ir(ir: Mapping[str, object]) -> dict[str, str]:
    constraints = ir.get("constraints", [])
    constraints = constraints if isinstance(constraints, list) else []
    properties = []
    directions: dict[str, str] = {}
    thresholds: dict[str, float] = {}
    instruction_tasks = []
    row = {
        "condition_id": str(ir.get("condition_id", "") or ""),
        "instruction": str(ir.get("instruction", "") or ""),
        "source_smiles": str(ir.get("source_smiles", "") or ""),
        "molecule_smiles": str(ir.get("source_smiles", "") or ""),
        "task_mode": "edit",
    }
    for item in constraints:
        if not isinstance(item, Mapping):
            continue
        prop = str(item.get("property", "") or "").strip()
        if not prop:
            continue
        properties.append(prop)
        direction_value = int(item.get("direction", 0) or 0)
        direction = "increase" if direction_value >= 0 else "decrease"
        directions[prop] = direction
        task = {"property": prop, "direction": direction}
        threshold = item.get("threshold")
        if threshold is not None:
            thresholds[prop] = float(threshold)
            task["threshold"] = float(threshold)
        instruction_tasks.append(task)
        source_value = item.get("source_value")
        if source_value is not None:
            row[f"external_source_{prop}"] = str(source_value)
        target = item.get("target")
        if target is not None:
            row[f"external_target_{prop}"] = str(target)
    row["condition_properties"] = ",".join(properties)
    row["external_task_properties"] = ",".join(properties)
    row["external_property_directions_json"] = json.dumps(directions, sort_keys=True)
    row["external_property_thresholds_json"] = json.dumps(thresholds, sort_keys=True)
    row["instruction_tasks"] = json.dumps(instruction_tasks, sort_keys=True)
    return row


def structural_action_key(value: object) -> tuple[object, ...] | None:
    if not isinstance(value, Mapping):
        return None
    bond = value.get("bond")
    if isinstance(bond, list):
        bond = tuple(bond)
    return (
        str(value.get("op", "") or ""),
        value.get("site"),
        bond,
        str(value.get("atom", "") or ""),
        str(value.get("fragment", "") or ""),
        str(value.get("bond_order", "") or ""),
    )


def action_payload(action: object) -> dict[str, object]:
    return {"action_type": "graph_edit_dsl", "value": asdict(action)}


def input_id_list(value: object) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Unsupported tokenizer output: {type(value).__name__}")
    return [int(item) for item in value]


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if int(left_id) != int(right_id):
            break
        count += 1
    return count


def encoded_action(
    tokenizer: object,
    prompt_messages: Sequence[Mapping[str, object]],
    payload: Mapping[str, object],
    *,
    max_length: int,
) -> dict[str, list[int]]:
    assistant = {"role": "assistant", "content": json.dumps(payload, sort_keys=True, separators=(",", ":"))}
    full_ids = input_id_list(
        tokenizer.apply_chat_template(
            [*prompt_messages, assistant],
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    prompt_ids = input_id_list(
        tokenizer.apply_chat_template(
            list(prompt_messages),
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and (not full_ids or full_ids[-1] != int(eos_id)):
        full_ids.append(int(eos_id))
    prompt_length = min(common_prefix_length(full_ids, prompt_ids), len(full_ids))
    assistant_ids = full_ids[prompt_length:]
    if len(assistant_ids) >= int(max_length):
        raise ValueError("Action target alone exceeds the scoring sequence length")
    if len(full_ids) > int(max_length):
        prompt_budget = int(max_length) - len(assistant_ids)
        head_budget = min(64, prompt_budget // 4)
        tail_budget = prompt_budget - head_budget
        prompt_prefix = full_ids[:prompt_length]
        compact_prompt = prompt_prefix[:head_budget] + prompt_prefix[-tail_budget:]
        full_ids = compact_prompt + assistant_ids
        prompt_length = len(compact_prompt)
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    if not any(value != -100 for value in labels):
        raise ValueError("Action target was truncated out of the scoring sequence")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def score_encoded_actions(
    model: object,
    tokenizer: object,
    encoded: Sequence[Mapping[str, Sequence[int]]],
    *,
    batch_size: int,
) -> list[float]:
    import torch
    import torch.nn.functional as functional

    scores = []
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(encoded), max(1, int(batch_size))):
        batch = encoded[start : start + max(1, int(batch_size))]
        width = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention = []
        labels = []
        for item in batch:
            padding = width - len(item["input_ids"])
            input_ids.append([*item["input_ids"], *([pad_id] * padding)])
            attention.append([*item["attention_mask"], *([0] * padding)])
            labels.append([*item["labels"], *([-100] * padding)])
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device="cuda")
        attention_tensor = torch.tensor(attention, dtype=torch.long, device="cuda")
        label_tensor = torch.tensor(labels, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            logits = model(input_ids=input_tensor, attention_mask=attention_tensor).logits[:, :-1, :]
        shifted_labels = label_tensor[:, 1:]
        token_losses = functional.cross_entropy(
            logits.transpose(1, 2),
            shifted_labels,
            reduction="none",
            ignore_index=-100,
        )
        mask = shifted_labels.ne(-100)
        sequence_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        scores.extend(float(-value) for value in sequence_losses.detach().cpu())
        del logits, input_tensor, attention_tensor, label_tensor
    return scores


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate_record(
    record: Mapping[str, object],
    *,
    model: object,
    tokenizer: object,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    policy = policy_module()
    ir, expected = constraint_payload(record)
    row = planner_row_from_ir(ir)
    candidates = policy.enumerate_action_candidates(
        row,
        site_limit=int(args.site_limit),
        max_actions_per_row=int(args.candidate_budget),
    )
    expected_key = structural_action_key(expected.get("value"))
    prompt_messages = record["messages"][:-1]
    encoded = [
        encoded_action(tokenizer, prompt_messages, action_payload(action), max_length=args.max_length)
        for action, _smiles, _program in candidates
    ]
    likelihoods = score_encoded_actions(
        model,
        tokenizer,
        encoded,
        batch_size=args.score_batch_size,
    ) if encoded else []
    ranked = sorted(
        [(*candidate, score) for candidate, score in zip(candidates, likelihoods)],
        key=lambda item: item[-1],
        reverse=True,
    )
    threshold = (
        args.table1_similarity_threshold
        if str(record.get("origin", "")) == "table1"
        else args.mumo_similarity_threshold
    )
    candidate_rows = []
    for rank, (action, smiles, program, score) in enumerate(ranked, start=1):
        oracle = policy.action_oracle_record(
            row,
            (action, smiles, program),
            source_similarity_threshold=float(threshold),
        )
        key = policy.action_key(action)
        candidate_rows.append(
            {
                "variant": args.variant,
                "example_id": record.get("example_id", ""),
                "origin": record.get("origin", ""),
                "rank": rank,
                "llm_mean_log_probability": score,
                "generated_smiles": smiles,
                "action_json": json.dumps(action_payload(action), sort_keys=True),
                "expected_action": key == expected_key,
                "strict_success": bool(oracle["strict_success"]),
                "instruction_success_fraction": oracle["instruction_success_fraction"],
                "source_similarity": oracle["source_similarity"],
                "source_similarity_success": oracle["source_similarity_success"],
            }
        )
    selected = candidate_rows[0] if candidate_rows else {}
    expected_ranks = [int(item["rank"]) for item in candidate_rows if item["expected_action"]]
    summary = {
        "variant": args.variant,
        "example_id": record.get("example_id", ""),
        "origin": record.get("origin", ""),
        "candidate_count": len(candidate_rows),
        "candidate_coverage": bool(candidate_rows),
        "expected_action_in_pool": bool(expected_ranks),
        "expected_action_at_1": expected_ranks == [1],
        "expected_action_at_20": bool(expected_ranks),
        "selected_strict_success": bool(selected.get("strict_success", False)),
        "any_strict_at_20": any(bool(item["strict_success"]) for item in candidate_rows),
        "selected_instruction_success_fraction": selected.get("instruction_success_fraction"),
        "selected_source_similarity": selected.get("source_similarity"),
        "selected_action_json": selected.get("action_json", ""),
        "selected_smiles": selected.get("generated_smiles", ""),
    }
    return summary, candidate_rows


def mean_bool(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / max(len(rows), 1)


def mean_finite(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [finite_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize(rows: Sequence[Mapping[str, object]], variant: str, budget: int) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    groups["all"] = list(rows)
    for row in rows:
        groups[str(row.get("origin", "unknown"))].append(row)
    return {
        "protocol": "unified_constraint_common_llm_constrained_action_v1",
        "variant": variant,
        "candidate_budget": budget,
        "groups": {
            name: {
                "rows": len(items),
                "candidate_coverage": mean_bool(items, "candidate_coverage"),
                "mean_candidate_count": mean_finite(items, "candidate_count"),
                "expected_action_at_1": mean_bool(items, "expected_action_at_1"),
                "expected_action_at_20": mean_bool(items, "expected_action_at_20"),
                "selected_strict_success": mean_bool(items, "selected_strict_success"),
                "any_strict_at_20": mean_bool(items, "any_strict_at_20"),
                "mean_selected_instruction_success_fraction": mean_finite(
                    items, "selected_instruction_success_fraction"
                ),
                "mean_selected_source_similarity": mean_finite(items, "selected_source_similarity"),
            }
            for name, items in sorted(groups.items())
        },
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_budget) != 20:
        raise ValueError("The paper-comparable constrained-action pilot requires candidate_budget=20")
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(f"Missing constrained-action evaluation dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("Constrained-action evaluation requires a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = True

    records = [row for row in read_jsonl(args.input_jsonl) if row.get("origin") in {"table1", "mumo"}]
    if args.max_rows > 0:
        records = records[: args.max_rows]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if args.adapter_dir:
        model = peft.PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.cuda().eval()

    output_rows = []
    candidate_rows = []
    for index, record in enumerate(records, start=1):
        output, candidates = evaluate_record(record, model=model, tokenizer=tokenizer, args=args)
        output_rows.append(output)
        candidate_rows.extend(candidates)
        print(f"[constrained-action] {index}/{len(records)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "details.csv", output_rows)
    write_csv(args.output_dir / "candidates.csv", candidate_rows)
    summary = summarize(output_rows, args.variant, args.candidate_budget)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
