#!/usr/bin/env python3
"""Evaluate one frozen adapter on an expanded P17 development view."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import p17_protocol as protocol


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completion_nll(model, tokenizer, messages: Sequence[Mapping[str, str]]) -> tuple[float, int]:
    import torch
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    answer = str(messages[-1]["content"]) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat((prompt_ids, answer_ids), dim=1).to(model.device)
    labels = ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    with torch.no_grad():
        loss = model(input_ids=ids, labels=labels).loss.float().item()
    return float(loss), int(answer_ids.numel())


def similarity(left: str, right: str) -> float | None:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    a, b = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
    if a is None or b is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return float(DataStructs.TanimotoSimilarity(gen.GetFingerprint(a), gen.GetFingerprint(b)))


def generate(model, tokenizer, messages, fixed_k: int, seed: int) -> tuple[str, list[str]]:
    import torch
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    offset = encoded["input_ids"].shape[1]
    with torch.no_grad():
        greedy_ids = model.generate(
            **encoded, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    greedy = tokenizer.decode(greedy_ids[0, offset:], skip_special_tokens=True).strip()
    torch.manual_seed(seed)
    with torch.no_grad():
        sampled_ids = model.generate(
            **encoded, max_new_tokens=128, do_sample=True, temperature=0.8, top_p=0.95,
            num_return_sequences=fixed_k, pad_token_id=tokenizer.pad_token_id,
        )
    sampled = [tokenizer.decode(ids[offset:], skip_special_tokens=True).strip() for ids in sampled_ids]
    return greedy, sampled


def scored(text: str, row: Mapping[str, object]) -> dict[str, object]:
    mode = str(row["task_mode"])
    parsed = protocol.parse_response(text, mode)
    predicted = str(parsed.get("smiles", ""))
    source = str(row["source_smiles"])
    is_edit = mode == "edit"
    return {
        "raw": text, "strict_parse": bool(parsed["strict_parse"]),
        "valid": bool(parsed["valid"]), "canonical": bool(parsed["canonical"]),
        "exact": bool(predicted and predicted == row["target_smiles"]),
        "noncopy": bool(predicted and predicted != source) if is_edit else None,
        "source_similarity": similarity(source, predicted) if is_edit and predicted else None,
        "smiles": predicted,
    }


def rate_block(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, float | int]:
    records = [row[key] for row in rows]
    out: dict[str, float | int] = {"rows": len(records)}
    for metric in ("strict_parse", "valid", "canonical", "exact", "noncopy"):
        supported = [item for item in records if item.get(metric) is not None]
        if supported:
            out[f"{metric}_rate"] = sum(bool(item[metric]) for item in supported) / len(supported)
    sims = [float(item["source_similarity"]) for item in records if item.get("source_similarity") is not None]
    if sims:
        out["mean_source_similarity"] = sum(sims) / len(sims)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--view", choices=("id", "ood"), required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--fixed-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1717)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers
    rows = read_jsonl(args.dev_jsonl)
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    loader = transformers.AutoModelForCausalLM if type(config) in transformers.AutoModelForCausalLM._model_mapping else transformers.AutoModelForImageTextToText
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = loader.from_pretrained(args.base_model, config=config, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True

    details = []
    for index, row in enumerate(rows):
        chosen_nll, chosen_tokens = completion_nll(model, tokenizer, row["messages"])
        copy_nll = None
        if row["task_mode"] == "edit":
            rejected_messages = [*row["messages"][:-1], {"role": "assistant", "content": row["rejected_assistant"]}]
            copy_nll, _ = completion_nll(model, tokenizer, rejected_messages)
        greedy_text, sample_texts = generate(model, tokenizer, row["messages"][:-1], args.fixed_k, args.seed + index)
        greedy = scored(greedy_text, row)
        candidates = [scored(text, row) for text in sample_texts]
        anyk = {
            "strict_parse": any(item["strict_parse"] for item in candidates),
            "valid": any(item["valid"] for item in candidates),
            "canonical": any(item["canonical"] for item in candidates),
            "exact": any(item["exact"] for item in candidates),
            "noncopy": any(item["noncopy"] for item in candidates) if row["task_mode"] == "edit" else None,
            "source_similarity": next((item["source_similarity"] for item in candidates if item["valid"]), None),
            "candidates": candidates,
        }
        details.append({
            "example_id": row["example_id"], "task_mode": row["task_mode"],
            "chosen_token_nll": chosen_nll, "chosen_tokens": chosen_tokens,
            "source_copy_token_nll": copy_nll,
            "chosen_minus_copy_nll": chosen_nll - copy_nll if copy_nll is not None else None,
            "greedy": greedy, "fixed_k": anyk,
        })
        if (index + 1) % 16 == 0 or index + 1 == len(rows):
            print(f"[p17-eval {args.label} {args.view}] {index + 1}/{len(rows)}", flush=True)

    metrics = {}
    for mode in ("de_novo", "edit"):
        subset = [row for row in details if row["task_mode"] == mode]
        token_total = sum(int(row["chosen_tokens"]) for row in subset)
        block = {
            "chosen_token_nll": sum(float(row["chosen_token_nll"]) * int(row["chosen_tokens"]) for row in subset) / max(token_total, 1),
            "greedy": rate_block(subset, "greedy"),
            f"any_at_{args.fixed_k}": rate_block(subset, "fixed_k"),
        }
        margins = [float(row["chosen_minus_copy_nll"]) for row in subset if row["chosen_minus_copy_nll"] is not None]
        if margins:
            block["mean_chosen_minus_copy_nll"] = sum(margins) / len(margins)
            block["chosen_preferred_to_copy_rate"] = sum(value < 0 for value in margins) / len(margins)
        metrics[mode] = block
    payload = {
        "protocol": "p17_expanded_raw_decode_v1", "label": args.label, "view": args.view,
        "adapter": str(args.adapter_dir), "fixed_k": args.fixed_k, "rows": len(rows),
        "selection": "none; raw generation order", "inference_target_access": False,
        "property_reranking": False, "metrics": metrics, "details": details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "details"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
