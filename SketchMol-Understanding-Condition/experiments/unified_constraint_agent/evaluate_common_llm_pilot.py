#!/usr/bin/env python3
"""Evaluate common-LLM JSON action control on train-only held-out records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_action_json(text: str) -> dict[str, object] | None:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else cleaned.strip("`")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def graph_action_module():
    script_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import build_external_graph_edit_agent_predictions

    return build_external_graph_edit_agent_predictions


def valid_smiles(value: object) -> bool:
    try:
        from rdkit import Chem
    except ImportError:
        return bool(str(value or "").strip())
    return Chem.MolFromSmiles(str(value or "")) is not None


def executable_graph_action(value: object, source_smiles: str) -> bool:
    if not isinstance(value, Mapping) or not source_smiles:
        return False
    graph = graph_action_module()
    fields = graph.GraphEditAction.__dataclass_fields__
    payload = {key: item for key, item in value.items() if key in fields}
    if isinstance(payload.get("bond"), list):
        payload["bond"] = tuple(payload["bond"])
    try:
        action = graph.GraphEditAction(**payload)
        generated = graph.execute_graph_edit_action(source_smiles, action)
    except (TypeError, ValueError, IndexError, KeyError):
        return False
    return valid_smiles(generated)


def score_output(row: Mapping[str, object], generated_text: str, variant: str) -> dict[str, object]:
    messages = row.get("messages", [])
    expected = json.loads(str(messages[-1]["content"]))
    user_payload = json.loads(str(messages[-2]["content"]))
    source = str(user_payload.get("constraint_ir", {}).get("source_smiles", "") or "")
    parsed = parse_action_json(generated_text)
    expected_type = str(expected.get("action_type", ""))
    predicted_type = str((parsed or {}).get("action_type", ""))
    value = (parsed or {}).get("value")
    if predicted_type == "smiles":
        executable = valid_smiles(value)
    elif predicted_type == "graph_edit_dsl":
        executable = executable_graph_action(value, source)
    else:
        executable = False
    return {
        "variant": variant,
        "example_id": row.get("example_id", ""),
        "origin": row.get("origin", ""),
        "task_mode": row.get("task_mode", ""),
        "expected_action_type": expected_type,
        "predicted_action_type": predicted_type,
        "json_parse_success": parsed is not None,
        "action_type_success": predicted_type == expected_type,
        "executable_action_success": executable,
        "exact_action_success": parsed == expected,
        "generated_text": generated_text,
    }


def mean_bool(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / max(len(rows), 1)


def summarize(rows: Sequence[Mapping[str, object]], variant: str) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    groups["all"] = list(rows)
    for row in rows:
        groups[str(row.get("origin", "unknown"))].append(row)
    return {
        "protocol": "unified_constraint_common_llm_format_eval_v1",
        "variant": variant,
        "groups": {
            name: {
                "rows": len(items),
                "json_parse_rate": mean_bool(items, "json_parse_success"),
                "action_type_rate": mean_bool(items, "action_type_success"),
                "executable_action_rate": mean_bool(items, "executable_action_success"),
                "exact_action_rate": mean_bool(items, "exact_action_success"),
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
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(f"Missing common-LLM evaluation dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("Common-LLM evaluation requires a CUDA GPU")

    records = read_jsonl(args.input_jsonl)
    if args.max_rows > 0:
        records = records[: args.max_rows]
    tokenizer_source = str(args.adapter_dir) if args.adapter_dir else args.base_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if args.adapter_dir:
        model = peft.PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.cuda().eval()
    output_rows = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                row["messages"][:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {key: value.cuda() for key, value in encoded.items()}
        with torch.inference_mode():
            sequences = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prefix_length = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(sequences[:, prefix_length:], skip_special_tokens=True)
        output_rows.extend(score_output(row, text, args.variant) for row, text in zip(batch, texts))
        print(f"[common-llm-eval] {len(output_rows)}/{len(records)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "details.csv", output_rows)
    with (args.output_dir / "details.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(output_rows, args.variant)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
