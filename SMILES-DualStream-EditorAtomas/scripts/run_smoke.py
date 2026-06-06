#!/usr/bin/env python3
"""Run a dependency-free smoke path for the pure-SMILES dual-stream project."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from smiles_dual_stream.data import SmilesPair, write_jsonl, write_summary  # noqa: E402
from smiles_dual_stream.featurize import build_dual_stream_example, summarize_examples  # noqa: E402


def main() -> int:
    pairs = [
        SmilesPair(sample_id="toy_self_001", source_smiles="CC(=O)O", target_smiles="CC(=O)O"),
        SmilesPair(sample_id="toy_edit_001", source_smiles="CCO", target_smiles="CCN", instruction="Replace alcohol with amine."),
        SmilesPair(sample_id="toy_edit_002", source_smiles="c1ccccc1O", target_smiles="c1ccccc1N", instruction="Edit phenol toward aniline."),
    ]
    examples = [build_dual_stream_example(pair, seed=11 + index) for index, pair in enumerate(pairs)]
    output_dir = PROJECT_DIR / "outputs" / "smoke"
    write_jsonl((example.to_dict() for example in examples), output_dir / "dual_stream_examples.jsonl")
    summary = summarize_examples(examples)
    write_summary(summary, output_dir / "summary.json")
    print(f"wrote smoke examples: {output_dir / 'dual_stream_examples.jsonl'}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

