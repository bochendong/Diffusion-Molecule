#!/usr/bin/env python3
"""Export unified description-pretraining and edit-generation JSONL datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sketchmol_unified_3m_diffusion.unified_condition_dataset import (  # noqa: E402
    read_3m_description_samples,
    read_edit_generation_samples,
    split_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--three-m-root",
        type=Path,
        default=REPO_DIR / "Research/Molecule Generation/3M-Diffusion",
    )
    parser.add_argument("--edit-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--description-limit-per-split", type=int, default=None)
    parser.add_argument("--edit-limit", type=int, default=None)
    parser.add_argument("--min-edit-source-tanimoto", type=float, default=None)
    parser.add_argument("--require-edit-quality-columns", action="store_true")
    parser.add_argument("--require-eval-oracle-strict", action="store_true")
    parser.add_argument("--include-pubchem", action="store_true")
    parser.add_argument("--include-kv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = []
    samples.extend(_read_description_bundle(args.three_m_root / "data/ChEBI-20_data", "ChEBI-20", args.description_limit_per_split))
    if args.include_pubchem:
        samples.extend(_read_description_bundle(args.three_m_root / "data/PubChem324k", "PubChem324k", args.description_limit_per_split))
    if args.include_kv:
        samples.extend(_read_description_bundle(args.three_m_root / "data/kv_data", "kv_data", args.description_limit_per_split))

    if args.edit_manifest is not None and args.edit_manifest.exists():
        samples.extend(
            read_edit_generation_samples(
                args.edit_manifest,
                limit=args.edit_limit,
                min_source_tanimoto=args.min_edit_source_tanimoto,
                require_quality_columns=args.require_edit_quality_columns,
                require_eval_oracle_strict=args.require_eval_oracle_strict,
            )
        )
    elif args.edit_manifest is not None:
        raise FileNotFoundError(f"edit manifest not found: {args.edit_manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "unified_condition_train.jsonl"
    eval_path = args.output_dir / "unified_condition_eval.jsonl"
    summary = split_samples(samples, train_output=train_path, eval_output=eval_path)
    summary.update(
        {
            "three_m_root": str(args.three_m_root),
            "edit_manifest": str(args.edit_manifest) if args.edit_manifest else None,
            "min_edit_source_tanimoto": args.min_edit_source_tanimoto,
            "require_edit_quality_columns": bool(args.require_edit_quality_columns),
            "require_eval_oracle_strict": bool(args.require_eval_oracle_strict),
            "train_jsonl": str(train_path),
            "eval_jsonl": str(eval_path),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _read_description_bundle(root: Path, dataset_name: str, limit: int | None) -> list:
    if not root.exists():
        return []
    specs = [
        ("train.txt", "train"),
        ("validation.txt", "eval"),
        ("valid.txt", "eval"),
        ("test_filter.txt", "eval"),
        ("test.txt", "eval"),
    ]
    samples = []
    seen = set()
    for filename, split in specs:
        path = root / filename
        if not path.exists() or path in seen:
            continue
        if filename == "test.txt" and (root / "test_filter.txt").exists():
            continue
        seen.add(path)
        samples.extend(
            read_3m_description_samples(path, split=split, dataset_name=dataset_name, limit=limit)
        )
    return samples


if __name__ == "__main__":
    main()
