#!/usr/bin/env python3
"""Preflight checks for unified 3M training jobs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-m-root", required=True, type=Path)
    parser.add_argument("--edit-manifest", required=True, type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--include-pubchem", action="store_true")
    parser.add_argument("--include-kv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {
        "three_m_root": str(args.three_m_root),
        "edit_manifest": str(args.edit_manifest),
        "require_cuda": bool(args.require_cuda),
    }

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit(f"Missing required Python package: {exc}") from exc

    report["numpy_version"] = np.__version__
    report["torch_version"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        report["cuda_device_count"] = int(torch.cuda.device_count())
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
    elif args.require_cuda:
        raise SystemExit("SMU3M_REQUIRE_CUDA=1 but torch.cuda.is_available() is false")

    report["three_m_files"] = _check_three_m_root(args.three_m_root, args.include_pubchem, args.include_kv)
    report["edit_manifest_first_usable_row"] = _check_edit_manifest(args.edit_manifest)
    print(json.dumps({"event": "preflight_ok", **report}, indent=2, sort_keys=True))


def _check_three_m_root(root: Path, include_pubchem: bool, include_kv: bool) -> dict[str, list[str]]:
    data_root = root / "data"
    if not data_root.is_dir():
        raise SystemExit(f"Missing 3M data directory: {data_root}")

    checks = {"ChEBI-20_data": ["train.txt", "validation.txt", "test.txt"]}
    if include_pubchem:
        checks["PubChem324k"] = ["train.txt"]
    if include_kv:
        checks["kv_data"] = ["train.txt"]

    found: dict[str, list[str]] = {}
    for dirname, filenames in checks.items():
        bundle = data_root / dirname
        if not bundle.is_dir():
            raise SystemExit(f"Missing 3M data bundle: {bundle}")
        present = [name for name in filenames if (bundle / name).is_file()]
        if not present:
            raise SystemExit(f"No expected split files found under {bundle}")
        found[dirname] = present
    return found


def _check_edit_manifest(path: Path) -> int:
    if not path.is_file():
        raise SystemExit(f"Missing edit manifest: {path}")
    if path.stat().st_size <= 0:
        raise SystemExit(f"Edit manifest is empty: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"source_smiles", "target_smiles"}
        missing = required - fieldnames
        if missing:
            raise SystemExit(f"Edit manifest missing columns: {sorted(missing)}")
        rows = 0
        for rows, row in enumerate(reader, start=1):
            if row.get("source_smiles") and row.get("target_smiles"):
                return rows
        raise SystemExit(f"Edit manifest has no usable source/target rows: {path}")


if __name__ == "__main__":
    main()
