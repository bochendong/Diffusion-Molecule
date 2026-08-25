#!/usr/bin/env python3
"""Generate exactly eight raw ordered candidates from one frozen P17 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import p17_protocol as protocol


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1717)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers
    rows = [json.loads(line) for line in args.prompts_jsonl.read_text().splitlines() if line.strip()]
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    loader = transformers.AutoModelForCausalLM if type(config) in transformers.AutoModelForCausalLM._model_mapping else transformers.AutoModelForImageTextToText
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = loader.from_pretrained(args.base_model, config=config, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True
    output = []
    for index, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        offset = encoded["input_ids"].shape[1]
        with torch.no_grad():
            greedy = model.generate(**encoded, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        torch.manual_seed(args.seed + index)
        with torch.no_grad():
            sampled = model.generate(
                **encoded, max_new_tokens=128, do_sample=True, temperature=0.8, top_p=0.95,
                num_return_sequences=7, pad_token_id=tokenizer.pad_token_id,
            )
        generated = [greedy[0], *list(sampled)]
        for candidate_index, ids in enumerate(generated):
            raw = tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
            parsed = protocol.parse_response(raw, row["task_mode"])
            output.append({
                "condition_id": row["condition_id"], "sample_id": row["sample_id"],
                "source_smiles": row["source_smiles"] if row["source_smiles"] != "<EMPTY>" else "",
                "generated_smiles": parsed.get("smiles", ""),
                "direct_candidate_canonical_smiles": parsed.get("smiles", ""),
                "direct_candidate_raw_smiles": raw,
                "direct_candidate_index": candidate_index,
                "candidate_rank": candidate_index + 1,
                "strict_parse": bool(parsed["strict_parse"]), "valid_smiles": bool(parsed["valid"]),
                "method": "p17_copy_contrastive_direct_llm",
            })
        print(f"[p17-pilot] {index + 1}/{len(rows)}", flush=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    summary = {
        "protocol": "p17_frozen_raw_pilot_generation_v1", "rows": len(rows),
        "candidates_per_row": 8, "candidate_zero": "greedy", "candidates_one_to_seven": "raw sampled",
        "inference_target_access": False, "static_candidate_pool": False,
        "property_reranking": False, "adapter": str(args.adapter_dir),
    }
    (args.output_csv.parent / f"{args.output_csv.stem}.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
