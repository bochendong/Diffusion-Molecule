#!/usr/bin/env python3
"""Generate one stochastic, unselected output for every frozen edit source."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import p23_protocol as protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=23501)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import peft
    import torch
    import transformers

    rows = [json.loads(line) for line in args.prompts_jsonl.read_text().splitlines() if line.strip()]
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
    tokenizer.padding_side = "left"
    base = loader.from_pretrained(
        args.base_model,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True

    output: list[dict[str, object]] = []
    batch_size = max(1, args.batch_size)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompt_text = [
            tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
            for row in batch
        ]
        encoded = tokenizer(prompt_text, return_tensors="pt", padding=True).to(model.device)
        offset = encoded["input_ids"].shape[1]
        torch.manual_seed(args.seed + start)
        with torch.no_grad():
            sampled = model.generate(
                **encoded,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row, ids in zip(batch, sampled):
            raw = tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
            parsed = protocol.parse_response(raw, row["task_mode"])
            output.append(
                {
                    "condition_id": row["condition_id"],
                    "sample_id": row["sample_id"],
                    "source_smiles": row["source_smiles"],
                    "generated_smiles": parsed.get("smiles", ""),
                    "direct_candidate_canonical_smiles": parsed.get("smiles", ""),
                    "direct_candidate_raw_smiles": raw,
                    "direct_candidate_index": 0,
                    "candidate_rank": 1,
                    "strict_parse": bool(parsed["strict_parse"]),
                    "valid_smiles": bool(parsed["valid"]),
                    "method": "p23_aligned24k_sampled_once",
                }
            )
        print(f"[p23-edit500] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {
        "protocol": "p23_moledit_table1_sampled_once_v1",
        "outputs": len(output),
        "outputs_per_source": 1,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "seed": args.seed},
        "greedy": False,
        "property_reranking": False,
        "target_access": False,
        "adapter": str(args.adapter_dir),
    }
    args.output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
