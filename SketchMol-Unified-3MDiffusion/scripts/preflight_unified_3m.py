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
    parser.add_argument("--edit-manifest", type=Path)
    parser.add_argument("--moledit-train-split", type=Path)
    parser.add_argument("--moledit-eval-split", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--include-pubchem", action="store_true")
    parser.add_argument("--include-kv", action="store_true")
    parser.add_argument("--min-edit-source-tanimoto", type=float, default=0.0)
    parser.add_argument("--require-edit-quality-columns", action="store_true")
    parser.add_argument("--require-eval-oracle-strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {
        "three_m_root": str(args.three_m_root),
        "edit_manifest": str(args.edit_manifest) if args.edit_manifest else None,
        "moledit_train_split": str(args.moledit_train_split) if args.moledit_train_split else None,
        "moledit_eval_split": str(args.moledit_eval_split) if args.moledit_eval_split else None,
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
    if args.edit_manifest is not None:
        report["edit_manifest"] = _check_edit_manifest(
            args.edit_manifest,
            min_source_tanimoto=args.min_edit_source_tanimoto,
            require_quality_columns=args.require_edit_quality_columns,
            require_eval_oracle_strict=args.require_eval_oracle_strict,
        )
    if args.moledit_train_split is not None:
        report["moledit_train_split"] = _check_moledit_split(
            args.moledit_train_split,
            split="train",
            min_source_tanimoto=args.min_edit_source_tanimoto,
        )
    if args.moledit_eval_split is not None:
        report["moledit_eval_split"] = _check_moledit_split(
            args.moledit_eval_split,
            split="eval",
            min_source_tanimoto=args.min_edit_source_tanimoto,
        )
    if args.edit_manifest is None and args.moledit_train_split is None and args.moledit_eval_split is None:
        raise SystemExit("Provide --edit-manifest or MolEdit split paths.")
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


def _check_edit_manifest(
    path: Path,
    *,
    min_source_tanimoto: float,
    require_quality_columns: bool,
    require_eval_oracle_strict: bool,
) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"Missing edit manifest: {path}")
    if path.stat().st_size <= 0:
        raise SystemExit(f"Edit manifest is empty: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"source_smiles", "target_smiles"}
        quality_columns = {
            "source_tanimoto",
            "source_similarity_bin",
            "pair_quality_tier",
            "strict_candidate_count_t04",
            "oracle_strict_success_t04",
            "preservation_constraint",
        }
        if require_quality_columns:
            required |= quality_columns
        missing = required - fieldnames
        if missing:
            raise SystemExit(f"Edit manifest missing columns: {sorted(missing)}")
        usable_rows = 0
        split_counts: dict[str, int] = {}
        first_usable_row = None
        low_source_tanimoto_rows = 0
        eval_oracle_failure_rows = 0
        missing_source_tanimoto_rows = 0
        min_seen_source_tanimoto = None
        for row_index, row in enumerate(reader, start=1):
            if not (row.get("source_smiles") and row.get("target_smiles")):
                continue
            usable_rows += 1
            if first_usable_row is None:
                first_usable_row = row_index
            split = row.get("split", "train") or "train"
            split_counts[split] = split_counts.get(split, 0) + 1
            source_tanimoto = _to_float(row.get("source_tanimoto", ""))
            if source_tanimoto is None:
                missing_source_tanimoto_rows += 1
                low_source_tanimoto_rows += 1
            else:
                min_seen_source_tanimoto = (
                    source_tanimoto
                    if min_seen_source_tanimoto is None
                    else min(min_seen_source_tanimoto, source_tanimoto)
                )
                if source_tanimoto < float(min_source_tanimoto):
                    low_source_tanimoto_rows += 1
            if split in {"eval", "valid", "validation", "test"} and _is_false(row.get("oracle_strict_success_t04", "")):
                eval_oracle_failure_rows += 1
        if usable_rows <= 0:
            raise SystemExit(f"Edit manifest has no usable source/target rows: {path}")
        if low_source_tanimoto_rows:
            raise SystemExit(
                "Edit manifest contains rows below the source-neighbor floor: "
                f"{low_source_tanimoto_rows} rows below {min_source_tanimoto}"
            )
        if require_eval_oracle_strict and eval_oracle_failure_rows:
            raise SystemExit(
                "Edit manifest contains eval rows without a source-neighbor strict candidate: "
                f"{eval_oracle_failure_rows} rows"
            )
        return {
            "path": str(path),
            "first_usable_row": first_usable_row,
            "usable_rows": usable_rows,
            "splits": split_counts,
            "min_source_tanimoto": min_seen_source_tanimoto,
            "missing_source_tanimoto_rows": missing_source_tanimoto_rows,
            "eval_oracle_failure_rows": eval_oracle_failure_rows,
            "required_quality_columns": bool(require_quality_columns),
            "required_eval_oracle_strict": bool(require_eval_oracle_strict),
        }


def _check_moledit_split(
    path: Path,
    *,
    split: str,
    min_source_tanimoto: float,
) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"Missing MolEdit {split} split: {path}")
    if path.stat().st_size <= 0:
        raise SystemExit(f"MolEdit {split} split is empty: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {
            "example_id",
            "instruction",
            "source_smiles",
            "target_smiles",
            "source_target_tanimoto",
            "instruction_tasks",
        }
        missing = required - fieldnames
        if missing:
            raise SystemExit(f"MolEdit {split} split missing columns: {sorted(missing)}")
        usable_rows = 0
        low_source_tanimoto_rows = 0
        missing_source_tanimoto_rows = 0
        min_seen_source_tanimoto = None
        task_counts: dict[str, int] = {}
        for row_index, row in enumerate(reader, start=1):
            if not (row.get("source_smiles") and row.get("target_smiles") and row.get("instruction")):
                continue
            usable_rows += 1
            source_tanimoto = _to_float(row.get("source_target_tanimoto", ""))
            if source_tanimoto is None:
                missing_source_tanimoto_rows += 1
            else:
                min_seen_source_tanimoto = (
                    source_tanimoto
                    if min_seen_source_tanimoto is None
                    else min(min_seen_source_tanimoto, source_tanimoto)
                )
                if source_tanimoto < float(min_source_tanimoto):
                    low_source_tanimoto_rows += 1
            task_props = row.get("instruction_task_properties", "") or row.get("computed_active_properties", "")
            task_counts[task_props or "unknown"] = task_counts.get(task_props or "unknown", 0) + 1
        if usable_rows <= 0:
            raise SystemExit(f"MolEdit {split} split has no usable source/target rows: {path}")
        if low_source_tanimoto_rows:
            raise SystemExit(
                "MolEdit split contains rows below the source-neighbor floor: "
                f"{low_source_tanimoto_rows} rows below {min_source_tanimoto}"
            )
        return {
            "path": str(path),
            "usable_rows": usable_rows,
            "split": split,
            "min_source_tanimoto": min_seen_source_tanimoto,
            "missing_source_tanimoto_rows": missing_source_tanimoto_rows,
            "task_properties": task_counts,
        }


def _to_float(value: object) -> float | None:
    try:
        return float(str(value if value is not None else "").strip())
    except ValueError:
        return None


def _is_false(value: object) -> bool:
    return str(value).strip().lower() in {"0", "false", "no", "n"}


if __name__ == "__main__":
    main()
