#!/usr/bin/env python3
"""Generate one target-hidden common-LLM repair plan per MuMO dev condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECT_SCRIPTS = PROJECT_DIR / "scripts"
for path in (SCRIPT_DIR, PROJECT_DIR, PROJECT_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_mumo_closed_loop_dev as closed_loop  # noqa: E402
import direct_repair_agent_protocol as repair  # noqa: E402
import evaluate_common_llm_pilot as common_eval  # noqa: E402
import mumo_parallel_protocol as protocol  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--dev-sources-jsonl", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=3)
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_digest(args: argparse.Namespace) -> str:
    payload = {
        "protocol": repair.PROTOCOL,
        "dev_sha256": file_sha256(args.dev_sources_jsonl),
        "adapter_sha256": file_sha256(args.adapter_dir / "adapter_model.safetensors"),
        "base_model": args.base_model,
        "max_steps": int(args.max_steps),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def read_progress(path: Path, digest: str) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    output = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if row.get("contract_digest") != digest:
            raise ValueError("Repair-plan checkpoint contract drift")
        output[str(row["condition_id"])] = dict(row)
    return output


def append_progress(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import peft
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("Direct repair planning requires CUDA")
    if not args.adapter_dir.joinpath("adapter_model.safetensors").is_file():
        raise FileNotFoundError(args.adapter_dir)
    torch.backends.cuda.matmul.allow_tf32 = True
    models = closed_loop.load_models(args.evidence_root)
    raw_rows = protocol.read_jsonl(args.dev_sources_jsonl)
    raw_rows.sort(key=lambda row: (str(row["_uca_task_id"]), str(row["_uca_source_group"])))
    digest = contract_digest(args)
    completed = read_progress(args.output_jsonl, digest)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = peft.PeftModel.from_pretrained(model, args.adapter_dir).cuda().eval()
    model.config.use_cache = True

    pending = []
    for index, raw in enumerate(raw_rows):
        row = closed_loop.condition_row(raw, index)
        condition_id = str(row["condition_id"])
        if condition_id in completed:
            continue
        source_feature = closed_loop.candidate_feature(str(row["source_smiles"]))
        if source_feature is None:
            raise ValueError(f"Invalid source feature: {condition_id}")
        properties = tuple(str(row["external_task_properties"]).split(","))
        _scores, margin_rows = closed_loop.score_candidates_batch(
            source_feature,
            [source_feature],
            properties=properties,
            models=models,
            source_tanimotos=[1.0],
            retrieval_similarities=[1.0],
            frequencies=[0],
        )
        margins = margin_rows[0]
        pending.append(
            {
                "condition_id": condition_id,
                "task_id": str(row["external_task_id"]),
                "properties": properties,
                "initial_margins": margins,
                "messages": repair.prompt_messages(
                    raw, properties, margins, max_steps=int(args.max_steps)
                ),
            }
        )

    for start in range(0, len(pending), int(args.batch_size)):
        batch = pending[start : start + int(args.batch_size)]
        prompts = [
            tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=True
            )
            for item in batch
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.cuda() for key, value in encoded.items()}
        with torch.inference_mode():
            sequences = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(args.max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prefix_length = encoded["input_ids"].shape[1]
        texts = tokenizer.batch_decode(sequences[:, prefix_length:], skip_special_tokens=True)
        for item, generated_text in zip(batch, texts):
            parsed = common_eval.parse_action_json(generated_text)
            order = repair.validate_plan(
                parsed,
                properties=item["properties"],
                max_steps=int(args.max_steps),
            )
            fallback = order is None
            if order is None:
                order = tuple(
                    sorted(
                        item["properties"],
                        key=lambda prop: (float(item["initial_margins"][prop]), prop),
                    )
                )
            record = {
                "condition_id": item["condition_id"],
                "task_id": item["task_id"],
                "property_order": list(order),
                "initial_train_verifier_margins": item["initial_margins"],
                "max_steps": int(args.max_steps),
                "controller_parse_success": not fallback,
                "controller_fallback_used": fallback,
                "generated_text": generated_text,
                "contract_digest": digest,
                "evaluation_target_access": False,
                "evaluation_oracle_access": False,
            }
            append_progress(args.output_jsonl, record)
            completed[str(item["condition_id"])] = record
        print(f"[direct-repair-plan] {len(completed)}/{len(raw_rows)}", flush=True)

    ordered = [completed[str(closed_loop.condition_row(raw, i)["condition_id"])] for i, raw in enumerate(raw_rows)]
    if len(ordered) != len(raw_rows):
        raise AssertionError("Repair-plan output is incomplete")
    parse_rate = sum(bool(row["controller_parse_success"]) for row in ordered) / max(len(ordered), 1)
    manifest = {
        "protocol": repair.PROTOCOL,
        "data_role": "fit_trained_controller_to_source_only_dev",
        "conditions": len(ordered),
        "plan_rows": len(ordered),
        "controller_parse_rate": parse_rate,
        "controller_fallback_rows": sum(bool(row["controller_fallback_used"]) for row in ordered),
        "max_steps": int(args.max_steps),
        "evaluation_target_access": False,
        "evaluation_oracle_access": False,
        "official_test_content_access": False,
        "output_selection": "none",
        "contract_digest": digest,
    }
    protocol.write_json(args.manifest_json, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
