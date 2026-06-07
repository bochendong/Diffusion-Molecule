#!/usr/bin/env python3
"""Prepare manifest and launch large-scale pure-SMILES dual-stream training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from smiles_dual_stream.config import get_section, load_config  # noqa: E402
from smiles_dual_stream.data import read_smiles_pairs, write_jsonl, write_summary  # noqa: E402
from smiles_dual_stream.featurize import build_dual_stream_example, summarize_examples  # noqa: E402
from smiles_dual_stream.train import main as train_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=PROJECT_DIR / "configs" / "large.yaml", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    _apply_env_overrides(config)
    paths = get_section(config, "paths")
    data = get_section(config, "data")
    train = get_section(config, "train")
    experiment = get_section(config, "experiment")

    input_csv = _resolve_existing_input(paths)
    manifest_jsonl = _repo_path(paths.get("manifest_jsonl", PROJECT_DIR / "outputs" / "manifests" / "large_train.jsonl"))
    summary_json = manifest_jsonl.with_suffix(".summary.json")
    train_output_dir = _repo_path(
        paths.get(
            "train_output_dir",
            PROJECT_DIR / "outputs" / "runs" / str(experiment.get("name", "large")),
        )
    )

    plan = {
        "config": str(args.config),
        "input_csv": str(input_csv),
        "manifest_jsonl": str(manifest_jsonl),
        "summary_json": str(summary_json),
        "train_output_dir": str(train_output_dir),
        "prepare": not args.train_only,
        "train": not args.prepare_only,
        "dry_run": args.dry_run,
        "train_config": train,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if not args.train_only:
        overwrite = bool(data.get("overwrite_manifest", False)) or args.overwrite_manifest
        if overwrite or not manifest_jsonl.exists():
            pairs = read_smiles_pairs(
                input_csv,
                source_column=str(data.get("source_column", "source_smiles")),
                target_column=str(data.get("target_column", "target_smiles")),
                smiles_column=str(data.get("smiles_column", "smiles")),
                instruction_column=str(data.get("instruction_column", "instruction")),
                id_column=str(data.get("id_column", "sample_id")),
                split_column=str(data.get("split_column", "split")),
                limit=_optional_int(data.get("limit")),
            )
            seed = int(data.get("seed", 7))
            examples = [build_dual_stream_example(pair, seed=seed + index) for index, pair in enumerate(pairs)]
            write_jsonl((example.to_dict() for example in examples), manifest_jsonl)
            write_summary(summarize_examples(examples), summary_json)
            print(f"prepared {len(examples)} examples: {manifest_jsonl}")
        else:
            print(f"reusing existing manifest: {manifest_jsonl}")

    if args.prepare_only:
        return 0

    import os

    train_args = [
        "--config",
        str(args.config),
        "--train-jsonl",
        str(manifest_jsonl),
        "--output-dir",
        str(train_output_dir),
    ]
    resume = bool(train.get("resume", True))
    resume_env = os.environ.get("SDEA_RESUME", "").strip().lower()
    if resume_env in {"0", "false", "no"}:
        resume = False
    elif resume_env in {"1", "true", "yes"}:
        resume = True
    train_args.append("--resume" if resume else "--no-resume")
    return train_main(train_args)


def _resolve_existing_input(paths: dict[str, object]) -> Path:
    input_csv = _repo_path(paths.get("input_csv", ""))
    if input_csv.exists():
        return input_csv
    fallback = _repo_path(paths.get("fallback_input_csv", ""))
    if fallback.exists():
        print(f"input_csv missing, using fallback_input_csv: {fallback}")
        return fallback
    raise FileNotFoundError(f"No input CSV found. Checked {input_csv} and {fallback}")


def _repo_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _apply_env_overrides(config: dict[str, object]) -> None:
    import os

    data = get_section(config, "data")
    train = get_section(config, "train")
    overrides = {
        "SDEA_LIMIT": (data, "limit", int),
        "SDEA_BATCH_SIZE": (train, "batch_size", int),
        "SDEA_EVAL_BATCH_SIZE": (train, "eval_batch_size", int),
        "SDEA_EMBED_DIM": (train, "embed_dim", int),
        "SDEA_HIDDEN_DIM": (train, "hidden_dim", int),
        "SDEA_EPOCHS": (train, "epochs", int),
        "SDEA_LR": (train, "lr", float),
        "SDEA_GRADIENT_ACCUMULATION_STEPS": (train, "gradient_accumulation_steps", int),
        "SDEA_MAX_SEQUENCE_LENGTH": (train, "max_sequence_length", int),
    }
    for env_name, (section, key, caster) in overrides.items():
        value = os.environ.get(env_name)
        if value is not None and value != "":
            section[key] = caster(value)


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
