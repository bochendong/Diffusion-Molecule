#!/usr/bin/env python3
"""Teacher-forced and raw-decoding audit for one P16 adapter."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import p16_protocol as protocol


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assistant_nll(model, tokenizer, row: Mapping[str, object]) -> tuple[float, int]:
    import torch

    prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
    answer = str(row["messages"][-1]["content"]) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = torch.cat((prompt_ids, answer_ids), dim=1).to(model.device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=labels).loss.float().item()
    return float(loss), int(answer_ids.numel())


def similarity(left: str, right: str) -> float:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    a, b = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
    if a is None or b is None:
        return math.nan
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(DataStructs.TanimotoSimilarity(generator.GetFingerprint(a), generator.GetFingerprint(b)))


def generation_texts(model, tokenizer, row: Mapping[str, object], fixed_k: int, seed: int) -> tuple[str, list[str]]:
    import torch

    prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    offset = encoded["input_ids"].shape[1]
    with torch.no_grad():
        greedy_ids = model.generate(**encoded, max_new_tokens=192, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    greedy = tokenizer.decode(greedy_ids[0, offset:], skip_special_tokens=True).strip()
    torch.manual_seed(seed)
    with torch.no_grad():
        sampled_ids = model.generate(
            **encoded, max_new_tokens=192, do_sample=True, temperature=0.8, top_p=0.95,
            num_return_sequences=fixed_k, pad_token_id=tokenizer.pad_token_id,
        )
    sampled = [tokenizer.decode(ids[offset:], skip_special_tokens=True).strip() for ids in sampled_ids]
    return greedy, sampled


def metric_block(details: Sequence[Mapping[str, object]], field: str) -> dict[str, float | int]:
    parsed = [item[field] for item in details]
    total = len(parsed)
    result: dict[str, float | int] = {"rows": total}
    for key in ("strict_parse", "valid", "canonical", "exact", "noncopy"):
        supported = [record for record in parsed if record.get(key) is not None]
        if supported:
            result[f"{key}_rate"] = sum(bool(record[key]) for record in supported) / len(supported)
    similarities = [float(record["source_similarity"]) for record in parsed if record.get("source_similarity") is not None]
    if similarities:
        result["mean_source_similarity"] = sum(similarities) / len(similarities)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--arm", choices=("mixed", "denovo", "edit"), required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--limit-per-mode", type=int, default=8)
    parser.add_argument("--fixed-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1616)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    all_rows = read_jsonl(args.dev_jsonl)
    allowed_modes = ("de_novo", "edit") if args.arm == "mixed" else (("de_novo",) if args.arm == "denovo" else ("edit",))
    rows: list[dict[str, object]] = []
    for mode in allowed_modes:
        rows.extend([row for row in all_rows if row["task_mode"] == mode][: args.limit_per_mode])
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    loader = transformers.AutoModelForCausalLM if type(config) in transformers.AutoModelForCausalLM._model_mapping else transformers.AutoModelForImageTextToText
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = loader.from_pretrained(args.base_model, config=config, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True

    nll_by_mode: dict[str, list[tuple[float, int]]] = defaultdict(list)
    details: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        mode = str(row["task_mode"])
        loss, tokens = assistant_nll(model, tokenizer, row)
        nll_by_mode[mode].append((loss, tokens))
        greedy_text, sampled_texts = generation_texts(model, tokenizer, row, args.fixed_k, args.seed + index)

        def scored(text: str) -> dict[str, object]:
            parsed = protocol.parse_response(text, mode)
            predicted = str(parsed.get("smiles", ""))
            target = str(row["target_smiles"])
            source = str(row["source_smiles"])
            is_edit = mode == "edit"
            sim = similarity(source, predicted) if is_edit and predicted else None
            return {
                "raw": text,
                "strict_parse": bool(parsed["strict_parse"]),
                "valid": bool(parsed["valid"]),
                "canonical": bool(parsed["canonical"]),
                "exact": bool(predicted and predicted == target),
                "noncopy": bool(predicted and predicted != source) if is_edit else None,
                "source_similarity": sim,
            }

        greedy = scored(greedy_text)
        candidates = [scored(text) for text in sampled_texts]
        fixed = {
            "strict_parse": any(item["strict_parse"] for item in candidates),
            "valid": any(item["valid"] for item in candidates),
            "canonical": any(item["canonical"] for item in candidates),
            "exact": any(item["exact"] for item in candidates),
            "noncopy": any(item["noncopy"] for item in candidates) if mode == "edit" else None,
            "source_similarity": next((item["source_similarity"] for item in candidates if item["valid"]), None),
            "candidates": candidates,
        }
        details.append({
            "example_id": row["example_id"], "task_mode": mode, "assistant_token_nll": loss,
            "assistant_tokens": tokens, "greedy": greedy, "fixed_k": fixed,
        })

    metrics: dict[str, object] = {}
    for mode in allowed_modes:
        subset = [item for item in details if item["task_mode"] == mode]
        values = nll_by_mode[mode]
        metrics[mode] = {
            "assistant_token_nll": sum(loss * tokens for loss, tokens in values) / max(sum(tokens for _, tokens in values), 1),
            "greedy": metric_block(subset, "greedy"),
            f"any_at_{args.fixed_k}": metric_block(subset, "fixed_k"),
        }
    payload = {
        "protocol": "p16_raw_direct_decode_eval_v1", "arm": args.arm, "fixed_k": args.fixed_k,
        "selection": "none; candidate metrics and any-in-generation-order only",
        "inference_target_access": False, "property_reranking": False, "rows": len(rows),
        "metrics": metrics, "details": details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "details"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
