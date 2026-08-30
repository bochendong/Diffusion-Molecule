#!/usr/bin/env python3
"""Build a larger direct-proposal train pool and reuse the frozen P32 gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
P32_DIR = SCRIPT_DIR.parent / "p32_unified_graph_repair_rl"
P30_DIR = SCRIPT_DIR.parent / "p30_balanced_shared_policy_rl"
P25_DIR = SCRIPT_DIR.parent / "p25_p23_joint_group_rl"
for path in (SCRIPT_DIR, P32_DIR, P30_DIR, P25_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import residual_protocol as protocol  # noqa: E402
import prepare_records as p32_prepare  # noqa: E402
import train_balanced_shared_rl as p30  # noqa: E402
import train_p23_joint_grpo as p25  # noqa: E402


def failure_summary(rows):
    failures = [row for row in rows if not bool(row["direct_details"].get("strict"))]
    return {
        "rows": len(rows),
        "direct_failures": len(failures),
        "failure_buckets": dict(Counter(str(row["bucket"]) for row in failures)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--p32-gate-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=32101)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    all_rows = p25.read_jsonl(args.train_jsonl)
    selected = []
    for mode, per_bucket, buckets in (
        ("de_novo", 20, p30.DE_NOVO_BUCKETS),
        ("edit", 12, p30.EDIT_BUCKETS),
    ):
        selected.extend(p32_prepare.select_balanced(
            [row for row in all_rows if str(row.get("task_mode")) == mode],
            buckets,
            per_bucket,
            args.seed + (0 if mode == "de_novo" else 1),
        ))

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
        args.base_model,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True
    prepared_train = p32_prepare.materialize(
        model, tokenizer, selected, batch_size=args.batch_size
    )
    prepared_gate = protocol.read_jsonl(args.p32_gate_jsonl)
    for row in [*prepared_train, *prepared_gate]:
        row["protocol"] = protocol.PROTOCOL
        row["initial_smiles"] = str(row.get("direct_smiles", "") or "") or "C"
        row["direct_details"] = protocol.direct_feedback(row).details

    by_mode_train = {
        mode: [row for row in prepared_train if row["task_mode"] == mode]
        for mode in ("de_novo", "edit")
    }
    by_mode_gate = {
        mode: [row for row in prepared_gate if row["task_mode"] == mode]
        for mode in ("de_novo", "edit")
    }
    if any(sum(not row["direct_details"].get("strict") for row in rows) < 30 for rows in by_mode_train.values()):
        raise ValueError({mode: failure_summary(rows) for mode, rows in by_mode_train.items()})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol.write_jsonl(args.output_dir / "train.jsonl", prepared_train)
    protocol.write_jsonl(args.output_dir / "gate.jsonl", prepared_gate)
    manifest = {
        "protocol": protocol.PROTOCOL,
        "train": {mode: failure_summary(rows) for mode, rows in by_mode_train.items()},
        "gate": {mode: failure_summary(rows) for mode, rows in by_mode_gate.items()},
        "p32_gate_conditions_and_proposals_reused": True,
        "direct_labels_recomputed_with_pinned_oracles": True,
        "target_molecules_used_for_policy_input": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
