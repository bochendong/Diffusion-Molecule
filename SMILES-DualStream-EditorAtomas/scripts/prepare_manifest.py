#!/usr/bin/env python3
"""Prepare pure-SMILES dual-stream JSONL examples from CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from smiles_dual_stream.data import read_smiles_pairs, write_jsonl, write_summary  # noqa: E402
from smiles_dual_stream.featurize import build_dual_stream_example, summarize_examples  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--source-column", default="source_smiles")
    parser.add_argument("--target-column", default="target_smiles")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--instruction-column", default="instruction")
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    pairs = read_smiles_pairs(
        args.input_csv,
        source_column=args.source_column,
        target_column=args.target_column,
        smiles_column=args.smiles_column,
        instruction_column=args.instruction_column,
        id_column=args.id_column,
        split_column=args.split_column,
        limit=args.limit,
    )
    examples = [build_dual_stream_example(pair, seed=args.seed + index) for index, pair in enumerate(pairs)]
    write_jsonl((example.to_dict() for example in examples), args.output_jsonl)
    summary = summarize_examples(examples)
    summary_path = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    write_summary(summary, summary_path)
    print(f"wrote {len(examples)} examples to {args.output_jsonl}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

