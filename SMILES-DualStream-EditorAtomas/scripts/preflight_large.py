#!/usr/bin/env python3
"""Preflight checks for pure-SMILES dual-stream large training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from smiles_dual_stream.config import get_section, load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = get_section(config, "paths")
    data = get_section(config, "data")
    train = get_section(config, "train")

    input_csv = _resolve_input(paths)
    usable_row = _check_csv(
        input_csv,
        source_column=str(data.get("source_column", "source_smiles")),
        target_column=str(data.get("target_column", "target_smiles")),
        smiles_column=str(data.get("smiles_column", "smiles")),
    )
    report: dict[str, object] = {
        "event": "preflight_ok",
        "config": str(args.config),
        "input_csv": str(input_csv),
        "first_usable_row": usable_row,
        "train_output_dir": str(_repo_path(paths.get("train_output_dir", ""))),
        "manifest_jsonl": str(_repo_path(paths.get("manifest_jsonl", ""))),
        "require_torch": bool(args.require_torch),
        "require_cuda": bool(args.require_cuda),
        "batch_size": train.get("batch_size"),
        "hidden_dim": train.get("hidden_dim"),
    }

    if args.require_torch or args.require_cuda:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(f"Missing required Python package: {exc}") from exc
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["cuda_device_count"] = int(torch.cuda.device_count())
            report["cuda_device_name"] = torch.cuda.get_device_name(0)
        elif args.require_cuda:
            raise SystemExit("SDEA_REQUIRE_CUDA=1 but torch.cuda.is_available() is false")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _resolve_input(paths: dict[str, object]) -> Path:
    input_csv = _repo_path(paths.get("input_csv", ""))
    if input_csv.is_file():
        return input_csv
    fallback = _repo_path(paths.get("fallback_input_csv", ""))
    if fallback.is_file():
        return fallback
    raise SystemExit(f"No input CSV found. Checked {input_csv} and {fallback}")


def _check_csv(path: Path, *, source_column: str, target_column: str, smiles_column: str) -> int:
    if not path.is_file():
        raise SystemExit(f"Missing input CSV: {path}")
    if path.stat().st_size <= 0:
        raise SystemExit(f"Input CSV is empty: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        has_pair = source_column in fieldnames and target_column in fieldnames
        has_single = smiles_column in fieldnames
        if not has_pair and not has_single:
            raise SystemExit(
                f"Input CSV must contain {source_column}/{target_column} or {smiles_column}; got {sorted(fieldnames)}"
            )
        for index, row in enumerate(reader, start=1):
            if has_pair and row.get(source_column) and row.get(target_column):
                return index
            if has_single and row.get(smiles_column):
                return index
    raise SystemExit(f"Input CSV has no usable SMILES rows: {path}")


def _repo_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())

