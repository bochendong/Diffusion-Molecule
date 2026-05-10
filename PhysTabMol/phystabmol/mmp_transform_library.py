"""CLI for mining a reusable MMP/CReM-style transform library."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .dataset import load_experiment_dataframe
from .mmp_transform_decoder import MMPTransformConfig, build_transform_library_dataframe


def main() -> None:
    args = parse_args()
    df = load_experiment_dataframe(
        data_path=args.data,
        smiles_column=args.smiles_column,
        image_column=None,
        limit=args.limit,
    )
    config = MMPTransformConfig(
        max_pairs=args.max_pairs,
        pairs_per_source=args.pairs_per_source,
        min_pair_similarity=args.min_pair_similarity,
        max_pair_similarity=args.max_pair_similarity,
        max_fragments=args.max_fragments,
        max_fragment_atoms=args.max_fragment_atoms,
    )
    library = build_transform_library_dataframe(df, config=config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    library.to_csv(out, index=False)
    print(f"Wrote {len(library)} transform-library rows -> {out}")
    if not library.empty and "record_type" in library:
        print(library["record_type"].value_counts().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine a reusable MMP/CReM-style transform library from a molecule CSV.")
    parser.add_argument("--data", default="data/molecules.csv")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--out", default="data/mmp_transform_library.csv")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--max-pairs", type=int, default=80000)
    parser.add_argument("--pairs-per-source", type=int, default=6)
    parser.add_argument("--min-pair-similarity", type=float, default=0.25)
    parser.add_argument("--max-pair-similarity", type=float, default=0.98)
    parser.add_argument("--max-fragments", type=int, default=12000)
    parser.add_argument("--max-fragment-atoms", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    main()
